from std.ffi import c_char
from std.memory import Pointer

struct MojoTensorBuffer(Copyable, Movable):
    var data_ptr: Pointer[UInt8, origin_of()]
    var num_elements: Int
    var element_size_bytes: Int
    var dtype: Int32

    def __init__(
        out self,
        data_ptr: Pointer[UInt8, origin_of()],
        num_elements: Int,
        element_size_bytes: Int,
        dtype: Int32,
    ):
        self.data_ptr = data_ptr
        self.num_elements = num_elements
        self.element_size_bytes = element_size_bytes
        self.dtype = dtype


struct MojoBlockTable(Copyable, Movable):
    var block_table_ptr: Pointer[Int32, origin_of()]
    var num_blocks: Int
    var block_size: Int

    def __init__(
        out self,
        block_table_ptr: Pointer[Int32, origin_of()],
        num_blocks: Int,
        block_size: Int,
    ):
        self.block_table_ptr = block_table_ptr
        self.num_blocks = num_blocks
        self.block_size = block_size


struct FfiResult(Copyable, Movable):
    var status_code: Int32
    var error_message: Optional[Pointer[c_char, origin_of()]]

    def __init__(
        out self,
        status_code: Int32,
        error_message: Optional[Pointer[c_char, origin_of()]] = None,
    ):
        self.status_code = status_code
        self.error_message = error_message

    @staticmethod
    def ok() -> Self:
        return Self(0, None)

    @staticmethod
    def err(
        status_code: Int32,
        error_message: Optional[Pointer[c_char, origin_of()]] = None,
    ) -> Self:
        return Self(status_code, error_message)
