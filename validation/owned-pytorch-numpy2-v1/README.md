# Shared NumPy 2 ABI build

This builds the same fourteen-file PyTorch source patch as owned half v3, against
NumPy 2.2.6. The C++/Metal source, Half storage checks, full-source inventory,
compiler flags and required MPS tests remain unchanged. Only the frozen NumPy
wheel changes in the 45-wheel dependency inventory.

The NumPy 1.26 joint run cannot complete the unchanged MLX Python inventory:
NumPy 2-only array APIs are required, and a padding reference also differs. The
old PyTorch build cannot simply load NumPy 2, so this is a complete hosted native
rebuild, never a local rebuild or a test/eligibility relaxation.

The transported native runtime must use NumPy 2.2.6 and demonstrate actual
`torch.from_numpy`/`Tensor.numpy` byte-preserving shared-memory roundtrips for
FP32, FP64, int32 and int64, including observed source mutation. Native source,
archive, dependency, origin, compiler-flag and FP16 bit/fenv checks remain required.
Hosted success establishes build/import/ABI evidence only. Both full framework
suites, the shared dependency inventory, model checks and every formal gate still
require fresh execution under the new native identity.

The workflow uses one hosted source/build tree, two compiler jobs, no debug
information and the existing allocation/free-space bounds. No local compilation
or replacement of an active runtime is authorized by this helper.
