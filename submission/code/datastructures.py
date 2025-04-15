class PC:

    _next_current_instruction = 0

    # Printable field
    _current_instruction = 0

    def __str__(self):
        return f"PC: {self._current_instruction}"

    def to_json(self):
        return self._current_instruction

    def get_pc(self):
        return self._current_instruction

    def set_pc(self, pc):
        self._next_current_instruction = pc

    def next_step(self):
        self._current_instruction = self._next_current_instruction

    def increment_and_get(self):
        ret = self._next_current_instruction
        self._next_current_instruction += 1
        return ret


class PhysicalRegisterFile:

    _num_physical_registers = 64
    _next_registers = [0] * _num_physical_registers

    # Printable field
    _registers = [0] * _num_physical_registers

    def __str__(self):
        return (
            f"PhysicalRegisterFile: [\n{', '.join(str(x) for x in self._registers)}\n]"
        )

    def to_json(self):
        return self._registers.copy()

    def next_step(self):
        self._registers = self._next_registers.copy()

    def pull_forwarding_path(self, register: int, value: int):
        self._registers[register] = value & 0xFFFFFFFFFFFFFFFF

    def set_register(self, register: int, value: int):
        self._next_registers[register] = value & 0xFFFFFFFFFFFFFFFF

    def get_register(self, register: int):
        return self._registers[register]


class DecodedInstructionRegister:

    _max_instructions = 4
    _next_decoded_pcs = []
    _next_added_pcs = []

    # Printable field
    _decoded_pcs = []

    def __str__(self):
        return f"DecodedPCs: [\n{', '.join(str(x) for x in self._decoded_pcs)}\n]"

    def to_json(self):
        return self._decoded_pcs.copy()

    def clear(self):
        self._next_decoded_pcs = []
        self._next_added_pcs = []

    def next_step(self):
        self._decoded_pcs = (self._next_decoded_pcs + self._next_added_pcs).copy()
        self._next_decoded_pcs = self._decoded_pcs.copy()
        self._next_added_pcs = []

    def pull_instructions(self, num_instructions: int):
        if num_instructions >= len(self._next_decoded_pcs):
            ret = self._next_decoded_pcs.copy()
            self._next_decoded_pcs = []
            return ret

        ret = self._next_decoded_pcs[:num_instructions].copy()
        self._next_decoded_pcs = self._next_decoded_pcs[num_instructions:].copy()
        return ret

    def get_free_space(self):
        return (
            self._max_instructions
            - len(self._next_decoded_pcs)
            - len(self._next_added_pcs)
        )

    def add_instruction(self, pc: int):
        self._next_added_pcs.append(pc)


class ExceptionUnit:

    # Printable field
    _exception_pc = 0
    _exception_status = False

    def __str__(self):
        return f"ExceptionPC: {self._exception_pc}\nException: {self._exception_status}"

    def exception_status(self):
        return self._exception_status

    def exception_pc(self):
        return self._exception_pc

    def start_exception(self, exception_pc):
        self._exception_pc = exception_pc
        self._exception_status = True

    def end_exception(self):
        self._exception_status = False


class RegisterMapTable:

    _num_architectural_registers = 32
    _next_register_map = list(range(32))

    # Printable field
    _register_map = list(
        range(32)
    )  # idx = architectural register / value = physical register

    def __str__(self):
        return (
            f"RegisterMapTable: [\n{', '.join(str(x) for x in self._register_map)}\n]"
        )

    def to_json(self):
        return self._register_map.copy()

    def next_step(self):
        self._register_map = self._next_register_map.copy()

    def set_register(self, architectural_register: int, physical_register: int):
        ret = self._next_register_map[architectural_register]
        self._next_register_map[architectural_register] = physical_register
        return ret

    def exception_set_register(
        self, architectural_register: int, physical_register: int
    ):
        ret = self._register_map[architectural_register]
        self._register_map[architectural_register] = physical_register
        return ret

    def get_register(self, architectural_register: int):
        return self._next_register_map[architectural_register]


class FreeList:

    _num_free_registers = 32
    _next_free_registers = []

    # Printable field
    _free_registers = []  # queue : append + pop(0)

    def __init__(self):
        self._free_registers = list(range(32, 32 + self._num_free_registers))
        self._next_free_registers = list(range(32, 32 + self._num_free_registers))

    def __str__(self):
        return f"FreeList: [\n{', '.join(str(x) for x in self._free_registers)}\n]"

    def to_json(self):
        return self._free_registers.copy()

    def check(self):
        if len(self._free_registers) > self._num_free_registers:
            return False
        return True

    def add_free_register(self, register: int):

        self._next_free_registers.append(register)

    def exception_add_free_register(self, register: int):
        self._free_registers = ([register] + self._free_registers).copy()

    def next_step(self):
        self._free_registers = self._next_free_registers.copy()

    def get_free_space(self):
        return self._num_free_registers - len(self._free_registers)

    def get_num_free_registers(self):
        return len(self._free_registers)

    def pull_free_register(self):
        if len(self._next_free_registers) == 0:
            return None
        return self._next_free_registers.pop(0)


class BusyBitTable:

    _num_registers = 64
    _next_busy_bits = [False] * _num_registers

    # Printable field
    _busy_bits = [False] * _num_registers

    def __str__(self):
        return f"BusyBitTable: [\n{', '.join(str(x) for x in self._busy_bits)}\n]"

    def to_json(self):
        return self._busy_bits.copy()

    def next_step(self):
        self._busy_bits = self._next_busy_bits.copy()

    def is_busy(self, register: int):
        return self._next_busy_bits[register]

    def set_busy(self, register: int):
        self._next_busy_bits[register] = True

    def clear_busy(self, register: int):
        self._next_busy_bits[register] = False

    def exception_clear_busy(self, register: int):
        self._busy_bits[register] = False

    def pull_forwarding_path(self, register: int):
        self._busy_bits[register] = False


class ActiveList:

    class ActiveListEntry:
        done: False
        exception: False
        architectural_destination_register: 0
        old_physical_destination_register: 0
        pc: 0

        def __init__(
            self,
            architectural_destination_register: int,
            old_physical_destination_register: int,
            pc: int,
        ):
            self.done = False
            self.exception = False
            self.architectural_destination_register = architectural_destination_register
            self.old_physical_destination_register = old_physical_destination_register
            self.pc = pc

        def to_json(self):
            return {
                "Done": self.done,
                "Exception": self.exception,
                "LogicalDestination": self.architectural_destination_register,
                "OldDestination": self.old_physical_destination_register,
                "PC": self.pc,
            }

    _max_active_instructions = 32
    _next_active_instructions = []
    _next_added_instructions = []

    # Printable field
    _active_instructions = []

    def to_json(self):
        return (
            []
            if self._active_instructions == []
            else [entry.to_json() for entry in self._active_instructions]
        )

    def check(self):
        if len(self._active_instructions) > self._max_active_instructions:
            return False
        return True

    def get_next_instruction(self):
        if self._active_instructions == []:
            return None

        return self._active_instructions[0]

    def get_num_next_done_instructions(self):
        if self._active_instructions == [] or not self._active_instructions[0].done:
            return 0

        for i in range(len(self._active_instructions)):
            if not self._active_instructions[i].done:
                return i  # i + 1 instr not done

        return len(self._active_instructions)

    def get_next_exception(self):
        if self._active_instructions == [] or not self._active_instructions[0].done:
            return -1

        for i in range(len(self._active_instructions)):
            if not self._active_instructions[i].done:
                return -1
            if self._active_instructions[i].exception:
                return i  # Start from 0

        return -1

    def get_instruction_by_pc(self, pc: int):
        for i in range(len(self._active_instructions)):
            if self._active_instructions[i].pc == pc:
                return self._active_instructions[i]
        return None

    def commit_instruction(self, nb_instructions: int):
        committed_instructions = self._next_active_instructions[:nb_instructions].copy()
        self._next_active_instructions = self._next_active_instructions[
            nb_instructions:
        ].copy()
        return committed_instructions

    def is_empty(self):
        return self._active_instructions is None or len(self._active_instructions) == 0

    def update_instruction(self, pc, exception):
        for i in range(len(self._next_active_instructions)):
            if self._next_active_instructions[i].pc == pc:
                self._next_active_instructions[i].exception = exception
                self._next_active_instructions[i].done = True

    def next_step(self):
        self._active_instructions = (
            self._next_active_instructions + self._next_added_instructions
        ).copy()
        self._next_active_instructions = self._active_instructions.copy()
        self._next_added_instructions = []

    def get_free_space(self):
        return self._max_active_instructions - len(self._next_active_instructions)

    def add_instruction(
        self,
        architectural_destination_register: int,
        old_physical_destination_register: int,
        pc: int,
    ):
        self._next_added_instructions.append(
            ActiveList.ActiveListEntry(
                architectural_destination_register,
                old_physical_destination_register,
                pc,
            )
        )

    def exception_next_step(self):
        ret_reversed_instructions = []
        while len(self._active_instructions) > 0 and len(ret_reversed_instructions) < 4:
            ret_reversed_instructions.append(self._active_instructions.pop())
        return ret_reversed_instructions

    def exception_first_step(self):
        return self._active_instructions.pop()


class IntegerQueue:

    class IntegerQueueEntry:
        dest_reg = 0
        op_a = (False, 0, 0)  # ready, reg_tag, value
        op_b = (False, 0, 0)  # ready, reg_tag, value
        op_code = "nop"
        pc = 0

        def __init__(
            self,
            dest_reg: int,
            op_a_reg: int,
            op_a_value: int | None,
            op_b_reg: int,
            op_b_value: int | None,
            op_code: str,
            pc: int,
        ):
            val_a = op_a_value if op_a_value is not None else 0
            val_b = op_b_value if op_b_value is not None else 0
            self.dest_reg = dest_reg
            self.op_a = (op_a_value is not None, op_a_reg, val_a)
            self.op_b = (op_b_value is not None, op_b_reg, val_b)
            self.op_code = op_code
            self.pc = pc

        def to_json(self):

            op_code_log = "add" if self.op_code == "addi" else self.op_code

            return {
                "DestRegister": self.dest_reg,
                "OpAIsReady": self.op_a[0],
                "OpARegTag": self.op_a[1],
                "OpAValue": self.op_a[2],
                "OpBIsReady": self.op_b[0],
                "OpBRegTag": self.op_b[1],
                "OpBValue": self.op_b[2],
                "OpCode": op_code_log,
                "PC": self.pc,
            }

        def __str__(self) -> str:
            return self.to_json()

        def is_ready(self):
            return self.op_a[0] and self.op_b[0]

    _max_instructions = 32
    _next_integer_queue = []
    _next_added_integer_queue = []

    # Printable field
    _integer_queue = []

    def to_json(self):
        return [entry.to_json() for entry in self._integer_queue]

    def __str__(self) -> str:
        entries = [str(entry.to_json()) for entry in self._integer_queue]
        return "[" + ", ".join(entries) + "]"

    def check(self):
        if len(self._integer_queue) > self._max_instructions:
            return False
        return True

    def clear(self):
        self._next_integer_queue = []

    def next_step(self):
        self._integer_queue = (
            self._next_integer_queue + self._next_added_integer_queue
        ).copy()
        self._next_integer_queue = self._integer_queue.copy()
        self._next_added_integer_queue = []

    def get_instruction_by_index(self, index: int):
        if index >= len(self._integer_queue) or index < 0:
            return None
        return self._integer_queue[index]

    def remove_instruction(self, pc: int):
        for i in range(len(self._next_integer_queue)):
            if self._next_integer_queue[i].pc == pc:
                self._next_integer_queue = (
                    self._next_integer_queue[:i] + self._next_integer_queue[i + 1 :]
                ).copy()
                return

    def pull_forwarding_path(self, dest_reg: int, value: int):
        for i in range(len(self._integer_queue)):
            if self._integer_queue[i].op_a[1] == dest_reg:
                self._integer_queue[i].op_a = (
                    True,
                    # self._integer_queue[i].op_a[1],
                    0,
                    value,
                )
            if self._integer_queue[i].op_b[1] == dest_reg:
                self._integer_queue[i].op_b = (
                    True,
                    # self._integer_queue[i].op_b[1],
                    0,
                    value,
                )
            if self._integer_queue[i].op_a[0] and self._integer_queue[i].op_b[0]:
                self._integer_queue[i].done = True

    def update_instruction(self, dest_reg: int, value: int):
        for i in range(len(self._next_integer_queue)):
            if self._next_integer_queue[i].op_a[1] == dest_reg:
                self._next_integer_queue[i].op_a = (
                    True,
                    self._next_integer_queue[i].op_a[1],
                    value,
                )
            if self._next_integer_queue[i].op_b[1] == dest_reg:
                self._next_integer_queue[i].op_b = (
                    True,
                    self._next_integer_queue[i].op_b[1],
                    value,
                )

    def get_free_space(self):
        return (
            self._max_instructions
            - len(self._next_integer_queue)
            - len(self._next_added_integer_queue)
        )

    def add_instruction(
        self,
        dest_reg: int,
        op_a_reg: int,
        op_a_value: int | None,
        op_b_reg: int | None,
        op_b_value: int | None,
        op_code: str,
        pc: int,
    ):
        if op_b_reg is None:  # imm case
            op_b_reg = 0

        if op_a_value is not None:  # don't care case
            op_a_reg = 0

        if op_b_value is not None:  # don't care case
            op_b_reg = 0

        self._next_added_integer_queue.append(
            IntegerQueue.IntegerQueueEntry(
                dest_reg, op_a_reg, op_a_value, op_b_reg, op_b_value, op_code, pc
            )
        )


class ALU:

    class ExecutedInstruction:
        dest_reg = 0
        op_a = (0, 0)  # reg_tag, value
        op_b = (0, 0)  # reg_tag, value
        op_code = "nop"
        pc = 0

        def __init__(
            self,
            instruction: IntegerQueue.IntegerQueueEntry,
        ):
            # TODO : Check that inputs are positive
            self.dest_reg = instruction.dest_reg
            self.op_a = (instruction.op_a[1], instruction.op_a[2])
            self.op_b = (instruction.op_b[1], instruction.op_b[2])
            self.op_code = instruction.op_code
            self.pc = instruction.pc

        def __str__(self):
            return f"ExecutedInstruction: {self.dest_reg}, {self.op_a}, {self.op_b}, {self.op_code}, {self.pc}"

        def is_exception(self):
            if self.op_code == "divu":
                return self.op_b[1] == 0
            elif self.op_code == "remu":
                return self.op_b[1] == 0
            return False

        def get_value_result(self):
            if self.op_code == "add":
                return (self.op_a[1] + self.op_b[1]) & 0xFFFFFFFFFFFFFFFF
            elif self.op_code == "addi":
                if self.op_b[1] & 0x80000000:  # A - imm = A + not(imm) + 1
                    return (
                        self.op_a[1] + (~self.op_b[1] & 0xFFFFFFFFFFFFFFFF) + 1
                    ) & 0xFFFFFFFFFFFFFFFF
                else:  # A + imm
                    return (self.op_a[1] + self.op_b[1]) & 0xFFFFFFFFFFFFFFFF
            elif self.op_code == "sub":
                if self.op_b[1] > self.op_a[1]:
                    return (
                        0xFFFFFFFFFFFFFFFF - (self.op_b[1] - self.op_a[1]) + 1
                    ) & 0xFFFFFFFFFFFFFFFF
                else:
                    return (self.op_a[1] - self.op_b[1]) & 0xFFFFFFFFFFFFFFFF
            elif self.op_code == "mulu":
                return (self.op_a[1] * self.op_b[1]) & 0xFFFFFFFFFFFFFFFF
            elif self.op_code == "divu":
                if self.op_b[1] == 0:
                    return -1
                return (self.op_a[1] // self.op_b[1]) & 0xFFFFFFFFFFFFFFFF
            elif self.op_code == "remu":
                if self.op_b[1] == 0:
                    return -1
                return (self.op_a[1] % self.op_b[1]) & 0xFFFFFFFFFFFFFFFF
            else:
                return -1

    _buffer: ExecutedInstruction | None = None
    _stage1: ExecutedInstruction | None = None
    _stage2: ExecutedInstruction | None = None

    def execute(self, instruction: IntegerQueue.IntegerQueueEntry):
        self._buffer = ALU.ExecutedInstruction(instruction)

    def next_step(self):
        self._stage2 = self._stage1
        self._stage1 = self._buffer
        self._buffer = None

    def get_result(self):
        return self._stage2
