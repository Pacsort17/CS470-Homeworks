from abc import abstractmethod
from enum import Enum


def str_to_int(str: str):
    if str.startswith("0x"):
        return int(str, 16)
    else:
        return int(str)


class InstructionType(Enum):
    add = "add"
    addi = "addi"
    sub = "sub"
    mulu = "mulu"
    ld = "ld"
    st = "st"
    loop = "loop"
    loop_pip = "loop.pip"
    nop = "nop"
    mov = "mov"


class DependenceType(Enum):
    local = "local"
    inter = "inter"
    inv = "inv"
    post = "post"


class Instruction:

    instruction_type: InstructionType
    pip_reg: int | None = None
    delay = 1
    block: int  # 0, 1 or 2
    pc: int

    def __init__(self, instruction_type, block, pc):
        self.instruction_type = InstructionType(instruction_type)
        self.block = block
        self.pc = pc

    def get_id(self):
        return self.pc

    def get_type(self):
        return self.instruction_type

    def get_delay(self):
        return self.delay

    def get_block_number(self):
        return self.block

    def update_block_number(self, new_block_number: int):
        self.block = new_block_number

    def set_pip_reg(self, new_pip_reg: int):
        self.pip_reg = new_pip_reg

    @abstractmethod
    def set_dest(self, new_dest: int):
        pass

    @abstractmethod
    def set_src(self, old_src: int, new_src: int):
        pass

    @abstractmethod
    def get_read_registers(self):
        pass

    @abstractmethod
    def get_dest(self):
        pass

    def __str__(self):
        if self.pip_reg is not None:
            return f"(p{self.pip_reg}) {self.instruction_type.value}"
        else:
            return f" {self.instruction_type.value}"


class InstructionALU(Instruction):
    old_opA: int | None = None
    old_opB: int | None = None

    def __init__(self, instruction_str, block, pc):
        instruction_str = instruction_str.strip().split(" ")
        super().__init__(instruction_str[0], block, pc)
        self.dest = instruction_str[1].lstrip("x").rstrip(",")
        self.opA = instruction_str[2].lstrip("x").rstrip(",")
        self.opB = (
            instruction_str[3].lstrip("x")
            if self.instruction_type != InstructionType.addi
            else None
        )
        self.imm = (
            instruction_str[3]
            if self.instruction_type == InstructionType.addi
            else None
        )
        if self.instruction_type == InstructionType.mulu:
            self.delay = 3

    def get_dest(self):
        return self.dest

    def get_opA(self):
        return self.opA

    def get_opB(self):
        return self.opB

    def get_imm(self):
        return self.imm

    def get_read_registers(self):
        return [self.opA, self.opB] if self.opB is not None else [self.opA]

    def set_dest(self, new_dest: int):
        self.dest = new_dest
        return True

    def set_src(self, old_src: int, new_src: int):
        if old_src == self.opA and self.old_opA is None:
            self.old_opA = self.opA
            self.opA = new_src
        if old_src == self.opB and self.old_opB is None:
            self.old_opB = self.opB
            self.opB = new_src

    def __str__(self):
        if self.opB is not None:
            return super().__str__() + f" x{self.dest}, x{self.opA}, x{self.opB}"
        else:
            return (
                super().__str__()
                + f" x{self.dest}, x{self.opA}, {str_to_int(self.imm)}"
            )


class InstructionMemory(Instruction):

    old_addr: int | None = None
    old_dest_source: int | None = None

    def __init__(self, instruction_str, block, pc):
        instruction_str = instruction_str.strip().split(" ")
        super().__init__(instruction_str[0], block, pc)
        self.dest_source = instruction_str[1].lstrip("x").rstrip(",")
        # Split imm(addr) format
        addr_str = instruction_str[2]
        imm_str, addr_str = addr_str.split("(")
        self.imm = imm_str
        self.addr = addr_str.lstrip("x").rstrip(")")

    def get_dest(self):
        return self.dest_source if self.instruction_type == InstructionType.ld else None

    def get_dest_source(self):
        return self.dest_source

    def get_imm(self):
        return self.imm

    def get_addr(self):
        return self.addr

    def get_read_registers(self):
        if self.instruction_type == InstructionType.ld:
            return [self.addr]
        else:
            return [self.dest_source, self.addr]

    # Works only for ld, use set_src for st
    def set_dest(self, new_dest: int):
        if self.instruction_type == InstructionType.ld:
            self.dest_source = new_dest
            return True
        return False

    def set_src(self, old_src: int, new_src: int):
        if old_src == self.addr and self.old_addr is None:
            self.old_addr = self.addr
            self.addr = new_src
        if (
            self.instruction_type == InstructionType.st
            and old_src == self.dest_source
            and self.old_dest_source is None
        ):
            self.old_dest_source = self.dest_source
            self.dest_source = new_src

    def __str__(self):
        return (
            super().__str__()
            + f" x{self.dest_source}, {str_to_int(self.imm)}(x{self.addr})"
        )


class InstructionLoop(Instruction):
    def __init__(self, instruction_str, block, pc):
        instruction_str = instruction_str.strip().split(" ")
        super().__init__(instruction_str[0], block, pc)
        self.imm_dest = instruction_str[1]

    def get_dest(self):
        return None

    def get_imm_dest(self):
        return self.imm_dest

    def get_read_registers(self):
        return []

    def set_dest_imm(self, new_dest: int):
        self.imm_dest = new_dest

    def set_dest(self, new_dest: str):
        return False

    def set_src(self, old_src: int, new_src: int):
        pass

    def __str__(self):
        return super().__str__() + f" {self.imm_dest}"


class InstructionNop(Instruction):
    def __init__(self, instruction_str, block, pc):
        super().__init__(InstructionType.nop, block, pc)

    def get_dest(self):
        return None

    def get_read_registers(self):
        return []

    def set_dest(self, new_dest: int):
        return False

    def set_src(self, old_src: int, new_src: int):
        pass

    def __str__(self):
        return super().__str__()


class InstructionMove(Instruction):
    def __init__(self, instruction_str, block, pc):
        super().__init__(InstructionType.mov, block, pc)
        instruction_str = instruction_str.strip().split(" ")
        self.dest = instruction_str[1].rstrip(",").lstrip("x").lstrip("p")
        self.src = (
            instruction_str[2].lstrip("x")
            if instruction_str[2].startswith("x")
            else None
        )
        self.imm = instruction_str[2] if self.src is None else None
        # 0 : mov pX, bool / 1 : mov LC-EC, imm / 2 : mov reg, imm / 3 : mov reg, reg
        self.mov_type = (
            3
            if self.src is not None
            else (
                0
                if len(self.imm) > 3 and (self.imm[0] == "t" or self.imm[0] == "f")
                else 1 if len(self.dest) == 2 and self.dest[1] == "C" else 2
            )
        )

    def get_dest(self):
        return self.dest

    def get_src(self):
        return self.src

    def get_read_registers(self):
        return [self.src] if self.src is not None else []

    def set_dest(self, new_dest: int):
        if self.mov_type > 1:
            self.dest = new_dest
            return True
        return False

    def set_src(self, old_src: int, new_src: int):
        if old_src == self.src:
            self.src = new_src

    def __str__(self):
        match self.mov_type:
            case 0:
                return super().__str__() + f" p{self.dest}, {self.imm}"
            case 1:
                return super().__str__() + f" {self.dest}, {str_to_int(self.imm)}"
            case 2:
                return super().__str__() + f" x{self.dest}, {str_to_int(self.imm)}"
            case 3:
                return super().__str__() + f" x{self.dest}, x{self.src}"


def parse_instruction(instruction_str, block, pc):
    if instruction_str.startswith("add"):
        return InstructionALU(instruction_str, block, pc)
    elif instruction_str.startswith("addi"):
        return InstructionALU(instruction_str, block, pc)
    elif instruction_str.startswith("sub"):
        return InstructionALU(instruction_str, block, pc)
    elif instruction_str.startswith("mulu"):
        return InstructionALU(instruction_str, block, pc)
    elif instruction_str.startswith("ld"):
        return InstructionMemory(instruction_str, block, pc)
    elif instruction_str.startswith("st"):
        return InstructionMemory(instruction_str, block, pc)
    elif instruction_str.startswith("loop"):
        return InstructionLoop(instruction_str, block, pc)
    elif instruction_str.startswith("loop.pip"):
        return InstructionLoop(instruction_str, block, pc)
    elif instruction_str.startswith("nop"):
        return InstructionNop(instruction_str, block, pc)
    elif instruction_str.startswith("mov"):
        return InstructionMove(instruction_str, block, pc)


class DependencyTable:

    class DependencyEntry:
        instr_addr: int
        instr_id: int
        instr_type: InstructionType
        dest_reg: int  # LC = -1, EC = -2

        # find operands from instruction ID

        def __init__(
            self, instruction_id: int, instruction_addr: int, instruction: Instruction
        ):
            self.instr_id = instruction_id  # position in the list of dependencies
            self.instr_addr = instruction_addr
            self.instr_type = instruction.get_type()
            self.dest_reg = instruction.get_dest()
            self.delay = instruction.get_delay()

            self.local_dep = dict()  # reg -> instr_id
            self.interloop_dep = dict()  # reg -> [instr_ids]
            self.loop_inv_dep = dict()  # reg -> instr_id
            self.post_loop_dep = dict()  # reg -> instr_id

        def __str__(self):
            return f"Instruction {self.instr_id} at {self.instr_addr}. Dependencies: local - {self.local_dep}, interloop - {self.interloop_dep}, loop_inv - {self.loop_inv_dep}, post_loop - {self.post_loop_dep}"

        def get_id(self):
            return self.instr_id

        def get_addr(self):
            return self.instr_addr

        def get_dest(self):
            return self.dest_reg

        def set_dest(self, new_dest: int):
            self.dest_reg = new_dest

        # returns the earliest pc where the result of the instruction is available
        def get_earliest_result_availability(self):
            # print(
            #    f"Instruction {self.instr_id} at {self.instr_addr} : earliest_result_availability = {self.instr_addr} + {self.delay}"
            # )
            return self.instr_addr + self.delay

        def set_addr(self, new_addr: int):
            self.instr_addr = new_addr

        def add_local_dependency(self, reg: int, waiting_instr_id: int):
            self.local_dep[reg] = (
                waiting_instr_id  # Check if new waiting_instr_id comes after current waiting_instr_id
            )

        def add_interloop_dependency(self, reg: int, waiting_instr_id: int):
            if reg not in self.interloop_dep:
                self.interloop_dep[reg] = list()
            self.interloop_dep[reg].append(waiting_instr_id)

        def add_loop_inv_dependency(self, reg: int, waiting_instr_id: int):
            self.loop_inv_dep[reg] = waiting_instr_id

        def add_post_loop_dependency(self, reg: int, waiting_instr_id: int):
            self.post_loop_dep[reg] = waiting_instr_id

        def get_local_dependencies_ids(self):
            return [id for id in self.local_dep.values()]

        def get_interloop_dependencies_ids(self):
            return [[i for i in ids] for ids in self.interloop_dep.values()]

        def get_loop_inv_dependencies_ids(self):
            return [id for id in self.loop_inv_dep.values()]

        def get_post_loop_dependencies_ids(self):
            return [id for id in self.post_loop_dep.values()]

        def get_dependencies(self):
            dep = list()
            for val in self.local_dep.values():
                dep.append((DependenceType.local, val))
            for val in self.interloop_dep.values():
                dep.append((DependenceType.inter, val))
            for val in self.loop_inv_dep.values():
                dep.append((DependenceType.inv, val))
            for val in self.post_loop_dep.values():
                dep.append((DependenceType.post, val))
            return dep

        def get_operand_with_dep(self):
            dep_operands = list()
            for key in self.local_dep.keys():
                dep_operands.append((DependenceType.local, key))
            for key in self.interloop_dep.keys():
                dep_operands.append((DependenceType.inter, key))
            for key in self.loop_inv_dep.keys():
                dep_operands.append((DependenceType.inv, key))
            for key in self.post_loop_dep.keys():
                dep_operands.append((DependenceType.post, key))
            return dep_operands

        def get_dependencies_origin_ids(self):
            return (
                self.local_dep
                | self.interloop_dep
                | self.loop_inv_dep
                | self.post_loop_dep
            )  # union of all dictionaries

    def __init__(self):
        self.dependencies = (
            list()
        )  # /!\ id of the instruction = position in the list /!\

    def __str__(self):
        return "\n".join([str(dep) for dep in self.dependencies])

    def get_instruction_dependencies(self, instruction_id: int):
        return self.dependencies[instruction_id]

    def get_earliest_free_pc(self, instruction_id: int):
        pc_res = 0
        instruction_dependencies = self.dependencies[instruction_id]
        for instr_id in instruction_dependencies.get_local_dependencies_ids():
            pc_res = max(
                pc_res, self.dependencies[instr_id].get_earliest_result_availability()
            )
        for instr_ids in instruction_dependencies.get_interloop_dependencies_ids():
            pc_res = max(
                pc_res,
                self.dependencies[min(instr_ids)].get_earliest_result_availability(),
            )  # either get from bb0 if there are two elements or from bb1 ( PC(bbo) < PC(bbl) )
        for instr_id in instruction_dependencies.get_loop_inv_dependencies_ids():
            pc_res = max(
                pc_res, self.dependencies[instr_id].get_earliest_result_availability()
            )
        for instr_id in instruction_dependencies.get_post_loop_dependencies_ids():
            pc_res = max(
                pc_res, self.dependencies[instr_id].get_earliest_result_availability()
            )
        return pc_res

    # Checks if instruction instruction_id with earliest_pc in bb1 depends on an instruction in bb0 (if earliest_pc is computed by a dependency in bb0)
    def check_bb1_bb0_dependency(self, instruction_id: int, earliest_pc: int):
        instruction_dependencies = self.dependencies[instruction_id]
        for instr_ids in instruction_dependencies.get_interloop_dependencies_ids():
            # If there is a interloop dependency with bb0, then it is in postion 0
            if (
                len(instr_ids) > 1
                and earliest_pc
                == self.dependencies[instr_ids[0]].get_earliest_result_availability()
            ):
                return True
        for instr_id in instruction_dependencies.get_loop_inv_dependencies_ids():
            if (
                earliest_pc
                == self.dependencies[instr_id].get_earliest_result_availability()
            ):
                return True
        return False

    def add_instruction(
        self,
        instruction: Instruction,
        instruction_id: int,
        instruction_addr: int,
        local_dependencies: dict[int, int],
        interloop_dependencies: dict[int, list[int]],
        loop_inv_dependencies: dict[int, int],
        post_loop_dependencies: dict[int, int],
    ):
        entry = DependencyTable.DependencyEntry(
            instruction_id, instruction_addr, instruction
        )

        for reg, dep in local_dependencies.items():
            entry.add_local_dependency(reg, dep)
        for reg, dep in interloop_dependencies.items():
            for d in dep:
                entry.add_interloop_dependency(reg, d)
        for reg, dep in loop_inv_dependencies.items():
            entry.add_loop_inv_dependency(reg, dep)
        for reg, dep in post_loop_dependencies.items():
            entry.add_post_loop_dependency(reg, dep)

        self.dependencies.append(entry)

    def update_address(self, instruction_id: int, new_addr: int):
        self.dependencies[instruction_id].set_addr(new_addr)  # /!\ id stays the same

    # Returns the addresses of the true interloop dependencies (PC(instruction) < PC(dependence)) + the corresponding delay
    def get_true_interloop_dependencies_earliest_free_address(
        self, instruction_id: int
    ):
        res = list()
        for ids in self.dependencies[instruction_id].get_interloop_dependencies_ids():
            for id in ids:
                if (
                    id >= instruction_id
                ):  # if id of the dependence is higher than instruction_id, it means PC(instruction) < PC(dependence), and thus instrcution is waiting for the result of the dependence in the previous cycle
                    res.append(self.dependencies[id].get_earliest_result_availability())
        return res


class BundleUnitType(Enum):
    alu = "alu"
    mul = "mul"
    branch = "branch"
    mem = "mem"
    anything = "any"


class Bundle:
    alu1: InstructionALU | InstructionMove | InstructionNop
    alu2: InstructionALU | InstructionMove | InstructionNop
    mult: InstructionALU | InstructionNop
    mem: InstructionMemory | InstructionNop
    branch: InstructionLoop | InstructionNop
    empty: bool

    def __init__(self):
        self.alu1 = None
        self.alu2 = None
        self.mult = None
        self.mem = None
        self.branch = None
        self.empty = True

    def get_unit(self, instr):
        if instr == self.alu1:
            return "alu1"
        elif instr == self.alu2:
            return "alu2"
        elif instr == self.mult:
            return "mul"
        elif instr == self.mem:
            return "mem"
        elif instr == self.branch:
            return "branch"
        else:
            return None

    def __str__(self):
        return (
            f"Bundle: {self.alu1}, {self.alu2}, {self.mult}, {self.mem}, {self.branch}"
        )

    def toJson(self):

        return [
            str(self.alu1) if self.alu1 is not None else " nop",
            str(self.alu2) if self.alu2 is not None else " nop",
            str(self.mult) if self.mult is not None else " nop",
            str(self.mem) if self.mem is not None else " nop",
            str(self.branch) if self.branch is not None else " nop",
        ]

    # Turn this object into an iterable
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < 5:
            self.index += 1
            if self.index == 1:
                return self.alu1
            elif self.index == 2:
                return self.alu2
            elif self.index == 3:
                return self.mult
            elif self.index == 4:
                return self.mem
            elif self.index == 5:
                return self.branch
        else:
            raise StopIteration

    def isEmpty(self):
        return self.empty

    # Returns True if the instruction was added, False if all space is occupied
    def add_entry(self, instruction: Instruction):
        added = True
        match instruction.get_type():
            case (
                InstructionType.add
                | InstructionType.addi
                | InstructionType.sub
                | InstructionType.mov
            ):
                if self.alu1 is None:
                    self.alu1 = instruction
                elif self.alu2 is None:
                    self.alu2 = instruction
                else:
                    added = False
            case InstructionType.mulu:
                if self.mult is None:
                    self.mult = instruction
                else:
                    added = False
            case InstructionType.loop | InstructionType.loop_pip:
                if self.branch is None:
                    self.branch = instruction
                else:
                    added = False
            case InstructionType.ld | InstructionType.st:
                if self.mem is None:
                    self.mem = instruction
                else:
                    added = False
        if self.empty:
            self.empty = not added
        return added

    def concatenate(self, other):
        for instr in other:
            if instr is None:
                continue
            added = self.add_entry(instr)
            if not added:
                print(f"Error concatenating {self} and {other}")
            self.empty = False


class Schedule:

    def __init__(self, dependency_table: DependencyTable):
        self.dependency_table = dependency_table
        self.bundles_bb0 = list()
        self.bundles_bb1 = list()
        self.bundles_bb2 = list()

    def toJson(self):
        return [
            bundle.toJson()
            for bundle in self.bundles_bb0 + self.bundles_bb1 + self.bundles_bb2
        ]

    def create_bundle(self, bb_number: int):
        if bb_number == 0:
            self.bundles_bb0.append(Bundle())
        elif bb_number == 1:
            self.bundles_bb1.append(
                Bundle()
            )  # No special handling (too hard to keep with add_instrcution) -> add loop instruction at the very end
        elif bb_number == 2:
            self.bundles_bb2.append(Bundle())

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        if (
            self.index
            < len(self.bundles_bb0) + len(self.bundles_bb1) + len(self.bundles_bb2) - 1
        ):
            self.index += 1
            if self.index < len(self.bundles_bb0):
                return self.bundles_bb0[self.index]
            elif self.index < len(self.bundles_bb0 + self.bundles_bb1):
                return self.bundles_bb1[self.index - len(self.bundles_bb0)]
            else:
                return self.bundles_bb2[
                    self.index - len(self.bundles_bb0) - len(self.bundles_bb1)
                ]
        else:
            raise StopIteration

    # Updates dependencies with the new address and returns the bundle where it was added /!\ No special case for loop instructions /!\
    # /!\ all instructions from bb0 must be added, then bb1 (except loop), then bb2 /!\
    def add_instruction(self, instruction: Instruction, instruction_id: int):

        if (
            instruction.get_type() == InstructionType.loop
            or instruction.get_type() == InstructionType.loop_pip
        ):
            print(
                f'Error: instruction "{instruction}" is a loop instruction and should be handled by another function'
            )
            return -1

        # Get the earliest free pc for the instruction
        earliest_free_pc = self.dependency_table.get_earliest_free_pc(instruction_id)
        # print(f"Instruction {instruction} : earliest_free_pc = {earliest_free_pc}")
        # /!\ Since they are 3 lists for bundles, earliest_free_pc is not the same for all lists /!\

        if instruction.get_block_number() == 0:
            while earliest_free_pc >= len(
                self.bundles_bb0
            ):  # creates the needed bundles to be able to add at bundle[earliest_free_pc]
                self.create_bundle(0)

            # Test if it can be added to the bundle, if not go to the next bundle
            while not self.bundles_bb0[earliest_free_pc].add_entry(instruction):
                earliest_free_pc += 1
                if earliest_free_pc >= len(
                    self.bundles_bb0
                ):  # if needed, creates a new bundle
                    self.create_bundle(0)

            # Updates dependencies with the new address
            self.dependency_table.update_address(instruction_id, earliest_free_pc)

            return earliest_free_pc
        elif instruction.get_block_number() == 1:
            relative_earliest_free_pc = max(
                0, earliest_free_pc - len(self.bundles_bb0)
            )  # if negative (available earlier) then consider it as 0
            # Handling bubbles : creates first needed bundle (when bb1 is empty), but then adds bundles to bb0 (and thus updates bb1 addresses)
            if len(self.bundles_bb1) == 0:
                self.create_bundle(1)
            # If the instruction depends on a bb0 instruction, then bubbles must be added
            if self.dependency_table.check_bb1_bb0_dependency(
                instruction_id, earliest_free_pc
            ):
                while relative_earliest_free_pc >= len(self.bundles_bb1):
                    self.create_bundle(0)
                    relative_earliest_free_pc -= 1
                    for i, bundle in enumerate(
                        self.bundles_bb1
                    ):  # ceci remporte la palme du code le moins optimisé du monde
                        for instr in bundle:
                            if instr is not None:
                                self.dependency_table.update_address(
                                    instr.get_id(),
                                    i + len(self.bundles_bb0),
                                )

            while relative_earliest_free_pc >= len(self.bundles_bb1):
                self.create_bundle(1)
            while not self.bundles_bb1[relative_earliest_free_pc].add_entry(
                instruction
            ):
                relative_earliest_free_pc += 1
                if relative_earliest_free_pc >= len(self.bundles_bb1):
                    self.create_bundle(1)

            # Recompute earliest_free_pc
            earliest_free_pc = relative_earliest_free_pc + len(self.bundles_bb0)
            self.dependency_table.update_address(instruction_id, earliest_free_pc)

            # Add enough bundles for interloop dependencies -> must be done after all instructions in bb1 are added

            return earliest_free_pc
        elif instruction.get_block_number() == 2:
            relative_earliest_free_pc = max(
                0, earliest_free_pc - len(self.bundles_bb0) - len(self.bundles_bb1)
            )
            while relative_earliest_free_pc >= len(self.bundles_bb2):
                self.create_bundle(2)  # adds bubbles in bb2

            while not self.bundles_bb2[relative_earliest_free_pc].add_entry(
                instruction
            ):
                relative_earliest_free_pc += 1
                if relative_earliest_free_pc >= len(self.bundles_bb2):
                    self.create_bundle(2)

            earliest_free_pc = (
                relative_earliest_free_pc
                + len(self.bundles_bb0)
                + len(self.bundles_bb1)
            )
            self.dependency_table.update_address(instruction_id, earliest_free_pc)
            return earliest_free_pc

        print(
            f'Error: instruction "{instruction}" not in a basic block, current block number: {instruction.get_block_number()}'
        )
        return -1

    def handle_interloop_dependencies(self):
        # adds bundles to bb1 until all interloop dependencies are handled
        for i, bundle in enumerate(
            self.bundles_bb1
        ):  # i = relative current address (relative to bb1)
            for instruction in bundle:
                if instruction is None:
                    continue
                dependencies_earliest_free_addresses = self.dependency_table.get_true_interloop_dependencies_earliest_free_address(
                    instruction.get_id()
                )
                for dep_free_addr in dependencies_earliest_free_addresses:
                    # While PC(instruction in next cycle) < PC(dependency), value is not ready (fictive, PC(instruction in next cycle) = PC(instruction) + length of loop)
                    while i + len(self.bundles_bb1) < dep_free_addr - len(
                        self.bundles_bb0
                    ):
                        self.create_bundle(1)

        # update all bb2 addresses
        for i, bundle in enumerate(self.bundles_bb2):
            for instruction in bundle:
                if instruction is not None:
                    self.dependency_table.update_address(
                        instruction.get_id(),
                        i + len(self.bundles_bb0) + len(self.bundles_bb1),
                    )

    def add_loop_instruction(self, instruction: InstructionLoop):
        if len(self.bundles_bb1) == 0:
            self.create_bundle(1)
        self.bundles_bb1[-1].add_entry(instruction)

    def get_start_loop_address(self):
        if len(self.bundles_bb1) == 0:
            return -1
        for bundle in self.bundles_bb1:
            for instruction in bundle:
                if (
                    instruction is not None
                    and instruction.get_type() != InstructionType.nop
                ):
                    return self.dependency_table.get_instruction_dependencies(
                        instruction.get_id()
                    ).get_addr()
        return -1

    def get_end_loop_address(self):
        if len(self.bundles_bb1) == 0:
            return -1
        for bundle in self.bundles_bb1:
            for instruction in bundle:
                if (
                    instruction is not None
                    and instruction.get_type() == InstructionType.loop
                ):
                    return instruction.get_addr()
        return -1

    def add_move_instruction(self, instruction: InstructionMove):
        if instruction.get_type() != InstructionType.mov:
            print(
                f'Error: When resolving WAW / WAR dependencies, instruction "{instruction}" is not a move instruction, current type: {instruction.get_type()}'
            )
            return -1

        relative_earliest_free_pc = max(
            (-1) * instruction.get_id() - len(self.bundles_bb0),
            len(self.bundles_bb1) - 1,
        )

        while relative_earliest_free_pc >= len(self.bundles_bb1):
            self.create_bundle(1)
        while not self.bundles_bb1[relative_earliest_free_pc].add_entry(instruction):
            self.create_bundle(1)
            relative_earliest_free_pc += 1
