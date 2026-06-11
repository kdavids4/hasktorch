{-# LANGUAGE CPP #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.GC where

import Control.Concurrent (threadDelay)
import Control.Concurrent.Async
import Control.Exception.Safe (Exception, MonadThrow, Typeable, bracket, catch, throwIO, throwM)
import Control.Monad (when)
import Data.List (isPrefixOf)
import Foreign.C.Types
import GHC.ExecutionStack
import Language.C.Inline.Cpp.Exception
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stderr)
import System.IO.Unsafe (unsafePerformIO)
import System.Mem (performGC)
import System.SysInfo
import qualified Data.Text.Encoding as T
import qualified Data.Text.Encoding.Error as T
import qualified Data.Text as T
import           Data.ByteString (ByteString)
import qualified Data.ByteString as B
import qualified Torch.Internal.Unmanaged.Type.Context as Context


foreign import ccall unsafe "hasktorch_finalizer.h showWeakPtrList"
  c_showWeakPtrList :: CInt -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h hasktorch_autorelease_pool_push"
  c_autorelease_pool_push :: IO ()

foreign import ccall unsafe "hasktorch_finalizer.h hasktorch_autorelease_pool_pop"
  c_autorelease_pool_pop :: IO ()

-- | Run an action with an Objective-C autorelease pool in scope. GHC
-- threads have no pool, and on macOS the Metal driver autoreleases buffer
-- wrapper objects while libtorch MPS ops execute; without a pool the objc
-- runtime leaks them — one Metal buffer per tensor, unbounded. Every
-- libtorch call dispatched through 'retryWithGC' is bracketed with this.
-- Pool tokens are tracked per OS thread on the C side, so a green-thread
-- migration between push and pop is harmless. No-op off macOS.
withAutoreleasePool :: IO a -> IO a
withAutoreleasePool func =
  bracket c_autorelease_pool_push (const c_autorelease_pool_pop) (const func)
{-# INLINE withAutoreleasePool #-}

-- malloc_trim is a glibc function. It doesn't exist on macos.
#ifdef ENABLE_DUMMY_MALLOC_TRIM
mallocTrim :: CInt -> IO ()
mallocTrim _ = return ()
#else
foreign import ccall unsafe "malloc.h malloc_trim"
  mallocTrim :: CInt -> IO ()
#endif

-- | Returns all objects of libtorch.
-- Each time it is called, the age of the object increases by one.
-- Dumps objects that are greater than or equal to the argument of age.
dumpLibtorchObjects ::
  -- | age
  Int ->
  -- | output
  IO ()
dumpLibtorchObjects age = c_showWeakPtrList (fromIntegral age)

newtype HasktorchException = HasktorchException String
  deriving (Show)

instance Exception HasktorchException

bsToChars :: ByteString -> String
bsToChars = T.unpack . T.decodeUtf8With T.lenientDecode

unsafeThrowableIO :: forall a m. MonadThrow m => IO a -> m a
unsafeThrowableIO a = unsafePerformIO $ (pure <$> a) `catch` (\(CppStdException _ msg _) -> pure . throwM $ HasktorchException ("Exception: " <> bsToChars msg))

prettyException :: IO a -> IO a
prettyException func =
  func `catch` \a@(CppStdException _ message _) -> do
    flag <- lookupEnv "HASKTORCH_DEBUG"
    when (flag /= Just "0") $ do
      mst <- showStackTrace
      case mst of
        Just st -> hPutStrLn stderr st
        Nothing -> hPutStrLn stderr "Cannot show stacktrace"
      B.hPutStr stderr message
    throwIO a
{-# INLINE prettyException #-}

retryWithGC' :: Int -> IO a -> IO a
retryWithGC' count func =
  func `catch` \a@(CppStdException _ message _) ->
    if B.isPrefixOf msgOutOfMemory message
      then
        if count <= 0
          then throwIO $ userError $ bsToChars $ "Too many calls to performGC, " <> message
          else do
            performGC
            mallocTrim 0
            threadDelay 1000 -- We need delta delay(1ms) to wait GC.
            -- GC only returns dead tensors' buffers to the MPS allocator's
            -- cache; the allocator's high-watermark check counts driver-level
            -- allocation including that cache, so a retry that needs a new
            -- heap can never succeed until the cache is flushed.
            Context.mps_empty_cache
            retryWithGC' (count -1) func
      else throwIO a
  where
#ifdef darwin_HOST_OS
    msgOutOfMemory = "MPS backend out of memory"
#else
    msgOutOfMemory = "CUDA out of memory."
#endif
{-# INLINE retryWithGC' #-}

retryWithGC :: IO a -> IO a
retryWithGC func = prettyException $ retryWithGC' 10 (withAutoreleasePool func)
{-# INLINE retryWithGC #-}

checkOSMemoryWithGC :: IO ()
checkOSMemoryWithGC = do
  v <- sysInfo
  case v of
    Right stat -> do
      let rate = (fromIntegral (freeram stat) / fromIntegral (totalram stat))
      if rate <= 0.5
        then do
          performGC
          mallocTrim 0
        else return ()
    Left _ -> return ()
  threadDelay (500 * 1000) -- wait 500msec
  checkOSMemoryWithGC

monitorMemory :: IO () -> IO ()
monitorMemory func = do
  func `race` checkOSMemoryWithGC
  return ()
