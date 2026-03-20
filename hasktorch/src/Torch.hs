module Torch
  ( module Torch,
    module Torch.Autograd,
    module Torch.Data,
    module Torch.Device,
    module Torch.DType,
    module Torch.Functional,
    module Torch.NN,
    module Torch.Optim,
    module Torch.Random,
    module Torch.Scalar,
    module Torch.Serialize,
    module Torch.Tensor,
    module Torch.TensorFactories,
    module Torch.TensorOptions,
    module Torch.Script,
    module Torch.Index,
    emptyMPSCache,
  )
where

import Torch.Autograd
import Torch.DType
import Torch.Data
import Torch.Device
import Torch.Functional
import Torch.Index
import Torch.NN
import Torch.Optim
import Torch.Random
import Torch.Scalar
import Torch.Script
import Torch.Serialize
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions
import qualified Torch.Internal.Managed.Type.Context as Context

-- | Flush the MPS (Metal Performance Shaders) device memory cache.
-- This releases cached memory blocks back to the OS, preventing
-- RSS growth from libtorch's MPS allocator pool.
-- No-op if MPS is not available.
emptyMPSCache :: IO ()
emptyMPSCache = Context.mps_empty_cache
