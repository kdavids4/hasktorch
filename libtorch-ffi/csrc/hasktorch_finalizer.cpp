#include "hasktorch_profile.h"
#include "hasktorch_finalizer.h"
#include <map>

#ifdef __APPLE__
// GHC finalizer/GC threads have no Objective-C autorelease pool. Freeing an
// MPS tensor releases its Metal buffer, and the AGX driver autoreleases
// wrapper objects in the process; with no pool in place the objc runtime
// "just leaks" them — one MTLBuffer per tensor, unbounded. Scope a pool
// around every finalizer so those releases actually happen.
//
// The pool functions are resolved with dlsym instead of linking libobjc:
// this file is linked into plain Haskell executables whose final link line
// has no -lobjc, but at runtime libobjc is always present in any process
// that loaded Metal-backed libtorch. Falls back to a no-op if absent.
#include <dlfcn.h>
namespace {
typedef void* (*ObjcPoolPushFn)(void);
typedef void (*ObjcPoolPopFn)(void*);
struct ObjcPoolFns {
  ObjcPoolPushFn push;
  ObjcPoolPopFn pop;
  ObjcPoolFns() {
    push = (ObjcPoolPushFn)dlsym(RTLD_DEFAULT, "objc_autoreleasePoolPush");
    pop = (ObjcPoolPopFn)dlsym(RTLD_DEFAULT, "objc_autoreleasePoolPop");
    if (!pop) push = nullptr;
  }
};
struct HasktorchAutoreleasePool {
  void* pool;
  static const ObjcPoolFns& fns() {
    static ObjcPoolFns f;
    return f;
  }
  HasktorchAutoreleasePool() : pool(fns().push ? fns().push() : nullptr) {}
  ~HasktorchAutoreleasePool() {
    if (pool) fns().pop(pool);
  }
};
}
#define HASKTORCH_FINALIZER_POOL HasktorchAutoreleasePool _hasktorch_pool;
#else
#define HASKTORCH_FINALIZER_POOL
#endif

// Exported so the Haskell side can bracket every libtorch FFI call with a
// pool: the AGX driver autoreleases Metal buffer objects during op
// execution/allocation on the calling thread, which for hasktorch is a
// pool-less GHC thread.
//
// Pool tokens are kept in a thread-local stack rather than handed back to
// Haskell: a green thread can migrate OS threads between push and pop, and
// popping a token on the wrong OS thread is undefined behavior in the objc
// runtime. With the TLS stack, pop always drains the *current* OS thread's
// most recent pool; a rare migration costs one missed drain, never a crash.
#ifdef __APPLE__
namespace {
thread_local std::vector<void*> hasktorch_pool_stack;
}
#endif

void hasktorch_autorelease_pool_push(){
#ifdef __APPLE__
  const ObjcPoolFns& f = HasktorchAutoreleasePool::fns();
  if (f.push) {
    hasktorch_pool_stack.push_back(f.push());
  }
#endif
}

void hasktorch_autorelease_pool_pop(){
#ifdef __APPLE__
  if (!hasktorch_pool_stack.empty()) {
    HasktorchAutoreleasePool::fns().pop(hasktorch_pool_stack.back());
    hasktorch_pool_stack.pop_back();
  }
#endif
}

void delete_tensor(at::Tensor* tensor){
  HASKTORCH_FINALIZER_POOL
  delete tensor;
}

void delete_optionaltensor(std::optional<at::Tensor>* tensor){
  HASKTORCH_FINALIZER_POOL
  delete tensor;
}

void delete_tensorlist(std::vector<at::Tensor>* tensors){
  HASKTORCH_FINALIZER_POOL
  delete tensors;
}

void delete_tensorindex(at::indexing::TensorIndex* idx){
  HASKTORCH_FINALIZER_POOL
  delete idx;
}

void delete_tensorindexlist(std::vector<at::indexing::TensorIndex>* idxs){
  HASKTORCH_FINALIZER_POOL
  delete idxs;
}

void delete_c10dict(c10::Dict<at::IValue,at::IValue>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10listivalue(c10::List<at::IValue>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10listtensor(c10::List<at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10listoptionaltensor(c10::List<c10::optional<at::Tensor>>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10listdouble(c10::List<double>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10listint(c10::List<int64_t>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10listbool(c10::List<bool>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdvectordouble(std::vector<double>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdvectorint(std::vector<int64_t>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdvectorbool(std::vector<bool>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_c10tuple(c10::intrusive_ptr<at::ivalue::Tuple>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_context(at::Context* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_dimname(at::Dimname* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_dimnamelist(std::vector<at::Dimname>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_generator(at::Generator* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_ivalue(at::IValue* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_ivaluelist(std::vector<at::IValue>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_intarray(std::vector<int64_t>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_module(torch::jit::script::Module* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_jitgraph(std::shared_ptr<torch::jit::Graph>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_jitnode(torch::jit::Node* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_jitvalue(torch::jit::Value* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_scalar(at::Scalar* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdarraybool2(std::array<bool,2>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdarraybool3(std::array<bool,3>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdarraybool4(std::array<bool,4>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stdstring(std::string* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_storage(at::Storage* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_symbol(at::Symbol* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensoroptions(at::TensorOptions* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensor(std::tuple<at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensorint64int64int64int64tensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensorlist(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensorint64(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensortensortensor(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensorcdoubleint64(std::tuple<at::Tensor,at::Tensor,double,int64_t>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensorint64int64tensor(std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensorint64int64tensortensor(std::tuple<at::Tensor,at::Tensor,int64_t,int64_t,at::Tensor,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensorlisttensor(std::tuple<std::vector<at::Tensor>,at::Tensor>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensorlist(std::tuple<at::Tensor,std::vector<at::Tensor>>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensortensorlisttensorlist(std::tuple<at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor>>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensorlisttensorlisttensorlisttensorlisttensorlist(std::tuple<std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_cdoubleint64(std::tuple<double,int64_t>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_cdoublecdouble(std::tuple<double,double>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_tensorgenerator(std::tuple<at::Tensor,at::Generator>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_optimizer(torch::optim::Optimizer* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_stream(c10::Stream* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_arrayrefscalar(at::ArrayRef<at::Scalar>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

void delete_vectorscalar(std::vector<at::Scalar>* object){
  HASKTORCH_FINALIZER_POOL
  delete object;
}

std::map<void*,int> objectAge;
std::map<void*,int> prevObjectAge;

void
shiftObjectMap(){
  prevObjectAge = objectAge;
  objectAge = std::map<void*,int>();
}

void
showObject(int flag, void* ptr, void* fptr){
  auto it = prevObjectAge.find(ptr);
  int age = 0;
  if (it != prevObjectAge.end()) {
    objectAge[ptr] = it->second + 1;
    age = it->second + 1;
  } else {
    objectAge[ptr] = 1;
    age = 1;
  }
  if(flag == 0)
    return;
  if(age < flag)
    return;
  if(fptr == (void*)delete_tensor){
    at::Tensor* t = (at::Tensor*) ptr;
    std::cout << age << ":" << "Tensor " << t->scalar_type() << " " << t->sizes() << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorlist){
    std::cout << age << ":" << "[Tensor]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorindex){
    std::cout << age << ":" << "tensorindex" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorindexlist){
    std::cout << age << ":" << "[tensorindex]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10dict){
    std::cout << age << ":" << "c10dict" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listivalue){
    std::cout << age << ":" << "c10listivalue" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listtensor){
    std::cout << age << ":" << "c10listtensor" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listoptionaltensor){
    std::cout << age << ":" << "c10listoptionaltensor" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listdouble){
    std::cout << age << ":" << "c10listdouble" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listint){
    std::cout << age << ":" << "c10listint" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10listbool){
    std::cout << age << ":" << "c10listbool" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdvectordouble){
    std::cout << age << ":" << "std::vector<double>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdvectorint){
    std::cout << age << ":" << "std::vector<int>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdvectorbool){
    std::cout << age << ":" << "std::vector<bool>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_c10tuple){
    std::cout << age << ":" << "c10tuple" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_context){
    std::cout << age << ":" << "context" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_dimname){
    std::cout << age << ":" << "dimname" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_dimnamelist){
    std::cout << age << ":" << "[dimname]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_generator){
    std::cout << age << ":" << "generator" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_ivalue){
    std::cout << age << ":" << "ivalue" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_ivaluelist){
    std::cout << age << ":" << "[ivalue]" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_intarray){
    std::cout << age << ":" << "intarray" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_module){
    std::cout << age << ":" << "module" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_jitgraph){
    std::cout << age << ":" << "jitgraph" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_jitnode){
    std::cout << age << ":" << "jitnode" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_jitvalue){
    std::cout << age << ":" << "jitvalue" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_scalar){
    std::cout << age << ":" << "scalar" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdarraybool2){
    std::cout << age << ":" << "std::array<bool,2>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdarraybool3){
    std::cout << age << ":" << "std::array<bool,3>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdarraybool4){
    std::cout << age << ":" << "std::array<bool,4>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stdstring){
    std::cout << age << ":" << "std::string" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_storage){
    std::cout << age << ":" << "storage" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_symbol){
    std::cout << age << ":" << "symbol" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensoroptions){
    std::cout << age << ":" << "tensoroptions" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensor){
    std::cout << age << ":" << "(tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensorint64int64int64int64tensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,int,int,int,int,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensorlist){
    std::cout << age << ":" << "(tensor,tensor,tensor,[tensor])" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensorint64){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor,int)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensortensortensor){
    std::cout << age << ":" << "(tensor,tensor,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorcdoubleint64){
    std::cout << age << ":" << "(tensor,tensor,double,int)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorint64int64tensor){
    std::cout << age << ":" << "(tensor,tensor,int,int,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorint64int64tensortensor){
    std::cout << age << ":" << "(tensor,tensor,int,int,tensor,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorlisttensor){
    std::cout << age << ":" << "(tensorlist,tensor)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorlist){
    std::cout << age << ":" << "(tensor,tensorlist)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensortensorlisttensorlist){
    std::cout << age << ":" << "(tensor,tensorlist,tensorlist)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorlisttensorlisttensorlisttensorlisttensorlist){
    std::cout << age << ":" << "(tensorlist,tensorlist,tensorlist,tensorlist,tensorlist)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_cdoubleint64){
    std::cout << age << ":" << "(double,int)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_cdoublecdouble){
    std::cout << age << ":" << "(double,double)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_tensorgenerator){
    std::cout << age << ":" << "(tensor,generator)" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_optimizer){
    std::cout << age << ":" << "optimizer" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_stream){
    std::cout << age << ":" << "stream" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_arrayrefscalar){
    std::cout << age << ":" << "at::ArrayRef<at::Scalar>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }else if(fptr == (void*)delete_vectorscalar){
    std::cout << age << ":" << "std::vector<at::Scalar>" << ":" << std::hex << (ptr) << std::dec << std::endl;
  }
}

