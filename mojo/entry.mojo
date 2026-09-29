from std.memory import Pointer
from mojo.types import FfiResult, MojoTensorBuffer


@export
def mojo_execute_kernel(
    input_ptr: Pointer[MojoTensorBuffer, origin_of()],
    output_ptr: Pointer[MojoTensorBuffer, origin_of()],
) abi("C") -> FfiResult:
    return FfiResult.ok()
