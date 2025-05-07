from datastructures import *
import json
import sys


def read_instructions(file_path):
    with open(file_path, "r") as f:
        instructions = json.load(f)
    return instructions


def write_log(json_data_list, file_path):
    with open(file_path, "w") as f:
        json.dump(json_data_list, f)


"""
Attention
La lecture de ce code peut entrainer des migraines ainsi que des nausées, 
et dans les cas les plus graves la perte totale des capacités cognitives.
Il est recommandé de se munir d'au moins 3 dolipranes et d'une dizaine de 
calmants, et d'entreprendre cette aventure accompagné, par un psy de préférence.

Merci pour votre compréhension,
Leander Cretegny (329736) et Roman Paccaud (327327)
"""


def main():
    input_path = sys.argv[1]
    # For test purposes
    if len(sys.argv) <= 2:
        output_path_loop = "given_tests/00/output_loop.json"
        output_path_loop_pip = "given_tests/00/output_loop_pip.json"
    else:
        output_path_loop = sys.argv[2]
        output_path_loop_pip = sys.argv[3]

    dependency_table = DependencyTable()
    dependency_table_pip = DependencyTable()

    raw_instructions = read_instructions(input_path)
    block_number = 2
    start_loop = -1  # imm contained in loop instruction
    end_loop = -1  # PC of loop instruction
    instructions = []
    instructions_pip = []
    raw_instructions = raw_instructions[::-1]
    # appending instructions from the last one, to determine the block number
    for i in range(len(raw_instructions)):
        current_pc = len(raw_instructions) - i - 1
        if raw_instructions[i].startswith("loop"):
            block_number = 1
            start_loop = int(raw_instructions[i].strip().split(" ")[1])
            end_loop = current_pc
        if current_pc == start_loop - 1:
            block_number = 0

        instructions.append(
            parse_instruction(raw_instructions[i], block_number, current_pc)
        )
        instructions_pip.append(
            parse_instruction(raw_instructions[i], block_number, current_pc)
        )

    # Handling no loop case
    if start_loop == -1:  # start_loop = end_loop = -1
        for instruction in instructions:
            instruction.update_block_number(0)
        for instruction in instructions_pip:
            instruction.update_block_number(0)

    instructions = instructions[
        ::-1
    ]  # reverse the instructions to have them in the correct order

    instructions_pip = instructions_pip[::-1]

    # Compute Initialisation Interval
    classes = [0, 0, 0, 0]
    units = [2, 1, 1, 1]
    for instruction in instructions[start_loop : end_loop + 1]:
        if instruction.block == 1:
            match instruction.get_type():
                case (
                    InstructionType.add
                    | InstructionType.addi
                    | InstructionType.sub
                    | InstructionType.mov
                ):
                    classes[0] += 1
                case InstructionType.mulu:
                    classes[1] += 1
                case InstructionType.ld | InstructionType.st:
                    classes[2] += 1
                case InstructionType.loop | InstructionType.loop_pip:
                    classes[3] += 1

    init_interval = 0
    for i in range(len(classes)):
        temp = (
            classes[i] // units[i]
            if classes[i] % units[i] == 0
            else classes[i] // units[i] + 1
        )  # manual ceiling function
        if temp > init_interval:
            init_interval = temp

    # Gather dependencies
    # Conseil : Prenez un doliprane avant de commencer à lire ce code
    for i, instruction in enumerate(instructions):
        local_dependencies = {}
        interloop_dependencies = {}
        loop_inv_dependencies = {}
        post_loop_dependencies = {}
        scan_instructions = []
        curr_bn = instruction.get_block_number()

        if curr_bn == 0:
            scan_instructions = instructions[: i + 1]
        elif curr_bn == 1:
            scan_instructions = instructions[
                : end_loop + 1
            ]  # taking last jump instruction
        else:
            scan_instructions = instructions[: i + 1]

        # BB1 : Start by scanning above instructions, then the one from previous block => [bb0, succ i, pred i] -> then reversed (local > interloop > invariant)
        if curr_bn == 1:
            scan_instructions = (
                scan_instructions[:start_loop]
                + scan_instructions[i:]
                + scan_instructions[start_loop:i]
            )
        scan_instructions = scan_instructions[
            ::-1
        ]  # scan in reverse to get last occurence

        """print(f"instruction {i} - {instruction} - {instruction.get_block_number()}")
        print(
            "\n".join([str(scan_instruction) for scan_instruction in scan_instructions])
        )
        print()"""

        read_regs = instruction.get_read_registers()
        if len(read_regs) > 1 and read_regs[0] == read_regs[1]:
            read_regs = [read_regs[0]]
        for read_reg in read_regs:
            for scan_instruction in scan_instructions:
                scan_i = scan_instruction.get_id()
                # i : PC addr of instruction -> id of instruction
                # scan_i : same but for scan_instruction
                if (
                    scan_instruction.get_dest() is not None
                    and scan_instruction.get_dest() == read_reg
                ):
                    scan_bn = scan_instruction.get_block_number()
                    if (
                        scan_bn == 1 and curr_bn == 1 and i <= scan_i
                    ):  # either bb0 -> bb1 or bb1 -> bb1 with PC(inst) < PC(scan)
                        # Scan block 0 -> interloop
                        if interloop_dependencies.get(read_reg) is not None:
                            continue  # if a dependency has already been found, skip
                        dep = [scan_i]
                        for k, instr_bb0 in enumerate(instructions[:start_loop][::-1]):
                            if instr_bb0.get_dest() == read_reg:
                                dep.append(start_loop - k - 1)  # id of instr_bb0
                                break

                        interloop_dependencies[read_reg] = dep[
                            ::-1
                        ]  # reverse the pair to have dependency from bb0 and the dependency from bb1
                    elif scan_bn == curr_bn:
                        if local_dependencies.get(read_reg) is not None or (
                            curr_bn != 1 and scan_i == i
                        ):
                            continue  # if a dependency has already been found, skip
                        local_dependencies[read_reg] = scan_i
                    elif (scan_bn == 0 and curr_bn == 2) or (
                        scan_bn == 0 and curr_bn == 1
                    ):  # works because we scan from the bottom (if curr_bn = 1 and scan_bn = 0, it means that there is no dependency from bb1 since bb1 got fully scanned)
                        if loop_inv_dependencies.get(read_reg) is not None:
                            continue  # if a dependency has already been found, skip
                        loop_inv_dependencies[read_reg] = scan_i
                    elif scan_bn == 1 and curr_bn == 2:
                        if post_loop_dependencies.get(read_reg) is not None:
                            continue  # if a dependency has already been found, skip
                        post_loop_dependencies[read_reg] = scan_i
                    else:
                        print(
                            f"Unhandled case: bb{curr_bn} -> bb{scan_bn} ({instruction} -> {scan_instruction})"
                        )
                    break  # dependency for this read_reg found, go to next reg

        dependency_table.add_instruction(
            instruction,
            i,
            i,  # At the beggining, addr = id
            local_dependencies,
            interloop_dependencies,
            loop_inv_dependencies,
            post_loop_dependencies,
        )

        dependency_table_pip.add_instruction(
            instructions_pip[i],
            i,
            i,
            local_dependencies,
            interloop_dependencies,
            loop_inv_dependencies,
            post_loop_dependencies,
        )

    """print(f"Initialisation Interval: {init_interval}")
    print(dependency_table)
    print()"""

    # Instruction added to the dependency table

    # ==================================================================================== #
    # Schedule loop
    loop_schedule = Schedule(dependency_table)  # from now on, only update in schedule

    for instruction in instructions:
        # skip loop instruction, must be added at the end
        if (
            instruction.get_type() == InstructionType.loop
            or instruction.get_type() == InstructionType.loop_pip
        ):
            continue
        loop_schedule.add_instruction(instruction, instruction.get_id())
    if start_loop > -1:
        loop_schedule.handle_interloop_dependencies()
        start_loop_addr = loop_schedule.get_start_loop_address()
        end_loop_addr = loop_schedule.get_end_loop_address()
        instructions[end_loop].set_dest_imm(start_loop_addr)
        # Add loop instruction after adding moves, to add it in the last bundle

    """print("--------------------------------")
    print()
    print("Before renaming:")
    print("Scheduler bb0:")
    for i, bundle in enumerate(loop_schedule.bundles_bb0):
        print(f"Bundle {i}: {bundle.toJson()}")
    print(f"Total bundles in bb0: {len(loop_schedule.bundles_bb0)}")
    print("\nScheduler bb1:")
    for i, bundle in enumerate(loop_schedule.bundles_bb1):
        print(f"Bundle {i}: {bundle.toJson()}")
    print(f"Total bundles in bb1: {len(loop_schedule.bundles_bb1)}")
    print("\nScheduler bb2:")
    for i, bundle in enumerate(loop_schedule.bundles_bb2):
        print(f"Bundle {i}: {bundle.toJson()}")
    print(f"Total bundles in bb2: {len(loop_schedule.bundles_bb2)}")
    print()
    print("--------------------------------")
    print()"""

    # Register renaming
    # 1) Rename all destinations
    free_reg = 1  # register number of the next free register
    for bundle in loop_schedule:
        for instruction in bundle:
            if instruction is not None and instruction.set_dest(str(free_reg)):
                dependency_table.get_instruction_dependencies(
                    instruction.get_id()
                ).set_dest(str(free_reg))
                free_reg += 1

    # 2) Rename sources (find dependencies, or wait step 4)
    no_dependencies = {}  # instruction id -> list of read regs without dependencies
    moves = (
        []
    )  # Instruction move, to add later (interloop dependencies that also have a dependency in bb0)
    already_moved = []
    for i, bundle in enumerate(loop_schedule):
        for instruction in bundle:
            if instruction is not None:
                no_dependencies[instruction.get_id()] = []
                dependencies = dependency_table.get_instruction_dependencies(
                    instruction.get_id()
                ).get_dependencies_origin_ids()  # dict reg -> dependency origin id, for all dependency (= not the ones without any dependency)
                for reg in instruction.get_read_registers():
                    if reg in dependencies.keys():
                        instruction_id = dependencies[reg]
                        if isinstance(instruction_id, list):
                            if (
                                len(instruction_id) > 1
                                and instruction_id[0] not in already_moved
                            ):
                                already_moved.append(instruction_id[0])
                                moves.append(
                                    InstructionMove(
                                        f"mov x{dependency_table.get_instruction_dependencies(instruction_id[0]).get_dest()}, x{dependency_table.get_instruction_dependencies(instruction_id[1]).get_dest()}",
                                        1,
                                        -(i + instruction.get_delay()),
                                    )  # pc set to the negative of the earliest address it can be scheduled, to not be mistaken for an id
                                )
                            instruction_id = instruction_id[0]  # either bb1 or bb0

                        # Set src to the dest of dependent instruction
                        instruction.set_src(
                            reg,
                            dependency_table.get_instruction_dependencies(
                                instruction_id
                            ).get_dest(),
                        )

                    else:
                        no_dependencies[instruction.get_id()].append(reg)

    # 3) Add move instructions for interloops
    for move in moves:
        loop_schedule.add_move_instruction(move)

    # Add loop instuction after adding moves, to add it in the last bundle
    if start_loop > -1:
        loop_schedule.add_loop_instruction(instructions[end_loop])

    # 4) Assign free registers to sources with no dependencies
    for bundle in loop_schedule:
        for instruction in bundle:
            if instruction is not None:
                if instruction.get_id() in no_dependencies.keys():
                    for reg in no_dependencies[instruction.get_id()]:
                        instruction.set_src(
                            reg,
                            str(free_reg),
                        )
                        free_reg += 1

    # End

    """
    print("====================================================")
    print()
    print("After renaming:")
    print("Scheduler bb0:")
    for i, bundle in enumerate(loop_schedule.bundles_bb0):
        print(f"Bundle {i}: {bundle.toJson()}")
    print(f"Total bundles in bb0: {len(loop_schedule.bundles_bb0)}")
    print("\nScheduler bb1:")
    for i, bundle in enumerate(loop_schedule.bundles_bb1):
        print(f"Bundle {i}: {bundle.toJson()}")
    print(f"Total bundles in bb1: {len(loop_schedule.bundles_bb1)}")
    print("\nScheduler bb2:")
    for i, bundle in enumerate(loop_schedule.bundles_bb2):
        print(f"Bundle {i}: {bundle.toJson()}")
    print(f"Total bundles in bb2: {len(loop_schedule.bundles_bb2)}")
    print()
    print("====================================================")
    write_log(loop_schedule.toJson(), output_path_loop)
    # End of loop
    # ==================================================================================== #

    print("--------------------------------")
    print()
    print("--------------------------------")
    """
    # ==================================================================================== #
    # Schedule pip #

    if start_loop == -1:
        start_loop = len(instructions_pip)

    schedule_pip = list()  # list[Bundle], the schedule
    bb0 = list()  # list[Bundle], the schedule for basic block 0
    bb1 = list()  # list[Bundle], the schedule for basic block 1
    bb2 = list()  # list[Bundle], the schedule for basic block 2

    is_valid = False  # Tracks validity of the current schedule
    II = init_interval
    while not is_valid:

        # Re-init schedule and validity
        bb0 = list()
        bb1 = list()
        bb2 = list()
        is_valid = True

        # Initialize table of reserved slots and stage counter
        reserved = {}  # Dict[unit -> List[Bundle addr]]
        reserved["alu1"] = list()
        reserved["alu2"] = list()
        reserved["mul"] = list()
        reserved["branch"] = list()
        reserved["mem"] = list()
        current_stage = 0

        # Iterate over all instruction and schedule them
        for i, instr in enumerate(instructions_pip):
            # print(f"scheduling instruction {i}")

            # Schedule loop instruction and fill basic block 1 if needed
            if instr.instruction_type == InstructionType.loop:

                # Scan for bundle full of nop at start of loop and put them at the end of basic block 0
                empty_bundles = list()
                for b in bb1:
                    if b.isEmpty():
                        empty_bundles.append(b)
                        bb0.append(Bundle())
                    else:
                        break
                for b in empty_bundles:
                    bb1.remove(b)

                while len(bb1) % II != 0:
                    bb1.append(Bundle())

                min_pc = II - 1
                # Loop instruction is added at the end of preparing the loop to have the final start of the loop
                # instr = InstructionLoop(f"loop.pip {len(bb0)}", 1, instr.pc)
                # bb1[min_pc].add_entry(instr)

            # Schedule non loop instruction
            elif instr.instruction_type != InstructionType.nop:
                # Schedule instruction in basic block 0
                if i < start_loop:
                    bb0, dependency_table_pip = schedule_pip_basic_block(
                        bb0, dependency_table_pip, instr, 0, 0
                    )
                # Schedule instruction in basic block 2
                elif i > end_loop:
                    bb2, dependency_table_pip = schedule_pip_basic_block(
                        bb2, dependency_table_pip, instr, 2, len(bb0) + len(bb1)
                    )
                # Schedule instruction in basic blocl 1 (the loop)
                else:
                    # Get dependencies
                    dep = dependency_table_pip.get_instruction_dependencies(
                        instr.pc
                    ).get_dependencies()

                    # Get earliest pc possible given dependencies (ignore inter loop for min pc computation if encountered),
                    # and if inter loop dependency encounter, add the dependency to inter loop dependency tracking table
                    inter_dep_tracking = {}  # Dict[ProducerId -> ConsumerId]
                    min_pc = 0
                    for dep_type, id in dep:
                        if dep_type is not DependenceType.inter:
                            free_slot = earliest_free_slot(bb1, instr)
                            dependence_available_from = (
                                dependency_table_pip.get_instruction_dependencies(
                                    id
                                ).get_earliest_result_availability()
                            )

                            # If dependency is a loop invariant, need to correctly compute pc from previous basic block
                            dependence_available_from = (
                                (
                                    0
                                    if dependence_available_from < len(bb0)
                                    else dependence_available_from - len(bb0)
                                )
                                if (dep_type == DependenceType.inv)
                                else dependence_available_from
                            )

                            min_pc = max(
                                min_pc,
                                (
                                    free_slot
                                    if free_slot >= dependence_available_from
                                    else dependence_available_from
                                ),
                            )
                        else:
                            inter_dep_tracking[max(id)] = instr.pc
                            free_slot = earliest_free_slot(bb1, instr)
                            dependence_available_from = (
                                dependency_table_pip.get_instruction_dependencies(
                                    min(id)
                                ).get_earliest_result_availability()
                            )
                            dependence_available_from = (
                                0
                                if dependence_available_from < len(bb0)
                                else dependence_available_from - len(bb0)
                            )
                            min_pc = max(
                                min_pc,
                                (
                                    free_slot
                                    if free_slot >= dependence_available_from
                                    else dependence_available_from
                                ),
                            )

                    # Check if min_pc is valid
                    if min_pc >= len(bb1):
                        bb1 = push_bundles(bb1, min_pc)
                    else:
                        while not is_bundle_slot_free(bb1[min_pc], instr):
                            min_pc += 1
                            if min_pc >= len(bb1):

                                bb1.append(Bundle())
                    # Check if slot is reserved
                    while is_reserved(min_pc, II, instr, reserved):
                        min_pc += 1
                        if min_pc >= len(bb1):
                            bb1.append(Bundle())
                    # Check if something depends on instr
                    if instr.pc in inter_dep_tracking.keys():
                        # Compute validity of schedule with current II
                        if instr.get_delay() > II:
                            is_valid = False
                        else:
                            producer_pc = min_pc % II
                            delay = instr.get_delay()
                            valid_consumer_pc = (delay + producer_pc) % II
                            consumer_pc = (
                                (
                                    dependency_table_pip.get_instruction_dependencies(
                                        inter_dep_tracking[instr.pc]
                                    ).get_addr()
                                    - len(bb0)
                                )
                                % II
                                if inter_dep_tracking[instr.pc] != instr.pc
                                else instr.pc
                            )
                            if valid_consumer_pc <= producer_pc:
                                is_valid = True
                            else:
                                is_valid = valid_consumer_pc <= consumer_pc

                    if not is_valid:
                        break
                    else:
                        # Schedule instruction
                        bb1[min_pc].add_entry(instr)
                        # Update reserved table
                        reserved[bb1[min_pc].get_unit(instr)].append(min_pc % II)
                        # Update address in dependency table
                        dependency_table_pip.update_address(instr.pc, min_pc)

            # End of for loop, pass to next instruction. Update current stage if needed
            if (i >= start_loop and i <= end_loop) and i % II == 0:
                current_stage += 1

        # All instruction scheduled, padd basic block with empty bundle to get len(bb1) % II == 0
        if not is_valid:
            # If schedule not valid retry with bigger II
            II += 1  # Retry with bigger initiation interval

    # End schedule pip #

    # Pip register allocation
    # Step 1 : Rename BB1 dest
    rotate_renaming = {}
    instr_new_dest = {}  # dict[instr.id -> renamed(instr.dest)]
    instr_source_renamed = {}  # dict[(instr.id, old_source_name) -> new source name]
    free_rotate_regs = list(range(32, 96))

    next_free_rotate_reg = 32
    for bundle in bb1:
        for instr in bundle:
            if instr is not None and instr.get_dest() is not None:
                try:
                    free_rotate_regs.remove(next_free_rotate_reg)
                except:
                    print(f"Error renaming {instr.get_dest()}")

                rotate_renaming[instr.get_dest()] = next_free_rotate_reg
                instr_new_dest[instr.get_id()] = next_free_rotate_reg
                instr.set_dest(next_free_rotate_reg)
                next_free_rotate_reg += len(bb1) // II + 1

    # Step 2 : Allocate loop invariant
    inv_renaming = {}
    free_inv_regs = list(range(1, 32))
    already_renamed = []

    # Loop on list of BB1 instruction and rename found loop invariant
    for b in bb1:
        for instr in b:
            if instr != None:
                dep_entry = dependency_table_pip.get_instruction_dependencies(
                    instr.get_id()
                )
                # Get dependency of current instruction
                dep = dep_entry.get_loop_inv_dependencies_ids()
                for d in dep:
                    # Get old name
                    old_name = dependency_table_pip.get_instruction_dependencies(
                        d
                    ).get_dest()
                    # Pop new name from free list
                    if old_name not in inv_renaming.keys():
                        new_name = free_inv_regs[0]
                        free_inv_regs.remove(new_name)
                        # Update map
                        inv_renaming[old_name] = new_name
                    # Rename in schedule
                    _, instr_new_dest = rename_in_bb(
                        bb0,
                        old_name,
                        inv_renaming[old_name],
                        instr_new_dest,
                        rename_dest=True,
                    )
                    already_renamed.append(old_name)
                """for reg in instr.get_read_registers():
                    if reg not in dep_entry.get_dependencies_origin_ids().keys():
                        new_name = free_inv_regs[0]
                        free_inv_regs.remove(new_name)
                        instr.set_src(reg, new_name)"""

    # Step 3 : Rename BB1 sources
    for b in bb1:
        for instr in b:
            if instr != None:
                dep_entry = dependency_table_pip.get_instruction_dependencies(
                    instr.get_id()
                )
                for dep, reg in dep_entry.get_operand_with_dep():
                    match dep:
                        case DependenceType.inv:
                            instr.set_src(reg, inv_renaming[reg])
                            instr_source_renamed[(instr.get_id(), reg)] = inv_renaming[
                                reg
                            ]
                        case DependenceType.local | DependenceType.inter:
                            producer_dep_entry = (
                                dependency_table_pip.get_instruction_dependencies(
                                    dep_entry.local_dep[reg]
                                    if dep == DependenceType.local
                                    else max(dep_entry.interloop_dep[reg])
                                )
                            )
                            producer_addr = producer_dep_entry.get_addr() + len(bb0)
                            st_producer = stage_of(len(bb0), II, producer_addr)
                            st_consumer = stage_of(
                                len(bb0), II, len(bb0) + dep_entry.get_addr()
                            )
                            new_name = instr_new_dest[producer_dep_entry.get_id()] + (
                                st_consumer - st_producer
                            )
                            if dep == DependenceType.inter:
                                new_name += 1
                            instr.set_src(reg, new_name)
                            instr_source_renamed[(instr.get_id(), reg)] = new_name

    # Step 4 : Allocate registers left in BB0 and BB2
    # Source register allocation
    for b in bb0:
        for instr in b:
            if instr != None:
                dep_entry: DependencyTable.DependencyEntry = (
                    dependency_table_pip.get_instruction_dependencies(instr.get_id())
                )

                if instr.get_dest() != None and instr.get_dest() != "LC":
                    reg = instr.get_dest()
                    reg_is_inter_dep_op, reg_producer, prod_bundle_addr = (
                        is_inter_dep_operand(
                            bb1, instr.get_id(), reg, dependency_table_pip
                        )
                    )
                    if reg_is_inter_dep_op and reg_producer != None:
                        # First case
                        producer = dependency_table_pip.get_instruction_dependencies(
                            reg_producer.get_id()
                        )
                        new_name = (
                            rotate_renaming[producer.get_dest()]
                            + 1
                            - stage_of(len(bb0), II, len(bb0) + prod_bundle_addr)
                        )
                        instr.set_dest(new_name)
                        instr_new_dest[instr.get_id()] = new_name
                    elif (
                        reg not in instr_new_dest.values()
                    ):  # reg not in rotate_renaming.keys() and
                        if reg not in inv_renaming.keys():
                            new_name = free_inv_regs.pop(0)
                            inv_renaming[reg] = new_name
                        # print(f"[DEBUG] renaming reg {reg} to {inv_renaming[reg]} in {instr}")
                        instr.set_dest(inv_renaming[reg])
                        instr_new_dest[instr.get_id()] = inv_renaming[reg]

                # Second case
                for reg in dep_entry.local_dep:
                    reg_dep_id = dep_entry.local_dep[reg]
                    # print(f"[DEBUG] renaming reg {reg} in {instr}, instr_new_dest: {instr_new_dest}")
                    instr.set_src(reg, instr_new_dest[reg_dep_id])
                    instr_source_renamed[(instr.get_id(), reg)] = instr_new_dest[
                        reg_dep_id
                    ]

                # Fourth case
                for reg in dep_entry.loop_inv_dep:
                    instr.set_src(reg, inv_renaming[reg])
                    instr_source_renamed[(instr.get_id(), reg)] = inv_renaming[reg]

                """# Fifth case - read registers
                for reg in instr.get_read_registers():
                    if reg not in dep_entry.get_dependencies_origin_ids().keys():
                        new_name = free_inv_regs[0]
                        free_inv_regs.remove(new_name)
                        instr.set_src(reg, new_name)"""

    for b in bb2:
        for instr in b:
            if instr != None:
                dep_entry: DependencyTable.DependencyEntry = (
                    dependency_table_pip.get_instruction_dependencies(instr.get_id())
                )

                if instr.get_dest() != None:
                    reg = instr.get_dest()
                    if (
                        reg not in instr_new_dest.values()
                    ):  # reg not in rotate_renaming.keys() and
                        if reg not in inv_renaming.keys():
                            new_name = free_inv_regs.pop(0)
                            inv_renaming[reg] = new_name
                        # print(f"[DEBUG] renaming reg {reg} to {inv_renaming[reg]} in {instr}")
                        instr.set_dest(inv_renaming[reg])
                        instr_new_dest[instr.get_id()] = inv_renaming[reg]

                # Second case
                for reg in dep_entry.local_dep:
                    reg_dep_id = dep_entry.local_dep[reg]
                    instr.set_src(
                        reg,
                        rotate_renaming[
                            dependency_table_pip.get_instruction_dependencies(
                                reg_dep_id
                            ).get_dest()
                        ],
                    )
                    instr_source_renamed[(instr.get_id(), reg)] = rotate_renaming[
                        dependency_table_pip.get_instruction_dependencies(
                            reg_dep_id
                        ).get_dest()
                    ]

                # Third case
                for reg in dep_entry.post_loop_dep:
                    reg_dep_id = dep_entry.post_loop_dep[reg]
                    producer_addr = dependency_table_pip.get_instruction_dependencies(
                        reg_dep_id
                    ).get_addr()
                    new_name = (
                        instr_new_dest[reg_dep_id]
                        + (len(bb1) // II)
                        - stage_of(len(bb0), II, len(bb0) + producer_addr)
                        - 1
                    )
                    instr.set_src(reg, new_name)
                    instr_source_renamed[(instr.get_id(), reg)] = new_name

                # Fourth case
                for reg in dep_entry.loop_inv_dep:
                    instr.set_src(reg, inv_renaming[reg])
                    instr_source_renamed[(instr.get_id(), reg)] = inv_renaming[reg]

    for b in bb0 + bb1 + bb2:
        for instr in b:
            if instr != None:
                dep_entry: DependencyTable.DependencyEntry = (
                    dependency_table_pip.get_instruction_dependencies(instr.get_id())
                )
                # Fifth case
                for reg in instr.get_read_registers():
                    if (
                        reg not in inv_renaming.values()
                        and reg not in rotate_renaming.values()
                        and reg not in instr_new_dest.values()
                        and reg not in instr_source_renamed.values()
                    ):
                        if (
                            str(reg)
                            not in dep_entry.get_dependencies_origin_ids().keys()
                        ):
                            new_name = free_inv_regs.pop(0)
                            """
                            print(
                                f"[DEBUG] renaming reg {reg} in {instr}, new name: {new_name}, free list is: {free_inv_regs}"
                                f"\ndependencies: {dep_entry.get_dependencies_origin_ids().keys()}"
                                f"\ninv map: {inv_renaming}"
                            )
                            """
                            instr.set_src(reg, new_name)

    # End pip register allocation #

    # Preparing the loop
    if len(bb1) > 0:
        nbr_stage = (len(bb1) // II) - 1
        # Step 1 & Step 2 : Sqeeze loop in one II and add predicate registers
        new_bb1 = list()
        first_pip_reg = 32
        for i in range(II):  # New bundle by new bundle
            new_bundle = Bundle()
            for j in range(i, len(bb1), II):  # Concatenate bundles
                for instr in bb1[j]:  # Step 2 : Add predicate registers
                    if instr is not None and instr.get_type() != InstructionType.loop:
                        instr.set_pip_reg(
                            first_pip_reg + ((j - i) // II)
                        )  # ((j - i) % II) is the pip reg number offset (current stage)
                new_bundle.concatenate(bb1[j])  # Step 1 : Concatenate bundles
            new_bb1.append(new_bundle)

        bb1 = new_bb1

        # Step 3 : Add mov instructions
        instr_predicate = InstructionMove(
            f"mov p{first_pip_reg}, true", 0, -1
        )  # PC not needed
        instr_EC = InstructionMove(f"mov EC, {nbr_stage}", 0, -1)  # PC not needed
        if len(bb0) == 0:
            bb0.append(Bundle())
        while not bb0[-1].add_entry(instr_EC):
            bb0.append(Bundle())
        while not bb0[-1].add_entry(instr_predicate):
            bb0.append(Bundle())

        # Add loop instruction
        bb1[len(bb1) - 1].add_entry(
            InstructionLoop(f"loop.pip {len(bb0)}", 1, -1)
        )  # PC not needed

    # End of loop.pip

    """
    print("====================================================")
    # Print schedule
    print_bb(bb0, 0)
    print_bb(bb1, 1)
    print_bb(bb2, 2)
    print("====================================================")
    """

    # Print dependency table
    json_bundles = [bundle.toJson() for bundle in bb0 + bb1 + bb2]
    write_log(json_bundles, output_path_loop_pip)

    # ==================================================================================== #


# ==================================================================================== #
# Schedule pip function #


def is_reserved(min_pc, II, instr, reserved):
    instr_unit = get_bundle_unit_type_of_instr(instr)
    target_bundle_pc = min_pc % II
    match instr_unit:
        case BundleUnitType.alu:
            if (
                target_bundle_pc in reserved["alu1"]
                and target_bundle_pc in reserved["alu2"]
            ):
                reserved = True
            else:
                reserved = False
        case BundleUnitType.mul:
            reserved = target_bundle_pc in reserved["mul"]
        case BundleUnitType.branch:
            reserved = target_bundle_pc in reserved["branch"]
        case BundleUnitType.mem:
            reserved = target_bundle_pc in reserved["mem"]
    return reserved


def print_bb(bb: list[Bundle], name):
    print(f"Basic block {name}: ")
    for i, b in enumerate(bb):
        print(f" Bundle {i}: {b.toJson()}")


def get_bundle_unit_type_of_instr(i: Instruction):
    match i.instruction_type:
        case (
            InstructionType.add
            | InstructionType.addi
            | InstructionType.sub
            | InstructionType.mov
        ):
            return BundleUnitType.alu
        case InstructionType.mulu:
            return BundleUnitType.mul
        case InstructionType.ld | InstructionType.st:
            return BundleUnitType.mem
        case InstructionType.loop | InstructionType.loop_pip:
            return BundleUnitType.branch
        case _:
            print(
                f"[ERROR] Unhandled instruction type {i.instruction_type} in get_bundle_unit_type_of_instr() for schedule pip"
            )
            return BundleUnitType.anything


def schedule_pip_basic_block(
    bb: list[Bundle],
    dependencies: DependencyTable,
    instr: Instruction,
    bb_number: int,
    start_of_bb: int,
):
    # Get dependencies
    dep = dependencies.get_instruction_dependencies(instr.pc).get_dependencies()

    # Get earliest pc possible given dependencies
    min_pc = 0
    for type, id in dep:
        free_slot = earliest_free_slot(bb, instr)
        dependence_available_from = dependencies.get_instruction_dependencies(
            id
        ).get_earliest_result_availability()
        dependence_available_from = (
            dependence_available_from - start_of_bb
            if bb_number == 2
            else dependence_available_from
        )
        min_pc = max(
            min_pc,
            (
                free_slot
                if free_slot >= dependence_available_from
                else dependence_available_from
            ),
        )

    # Check if slot at earliest pc is free, else if it is bigger than current size of schedule, else augment it until finding a free slot. When a valid spot is found add instruction to schedule
    if min_pc >= len(bb):
        bb = push_bundles(bb, min_pc)
    else:
        while not is_bundle_slot_free(bb[min_pc], instr):
            min_pc += 1
            if min_pc >= len(bb):
                bb.append(Bundle())
    bb[min_pc].add_entry(instr)

    # Update address in dependency table
    dependencies.update_address(instr.pc, min_pc)

    return bb, dependencies


def push_bundles(bb: list[Bundle], goal: int):
    while len(bb) <= goal:
        bb.append(Bundle())
    return bb


def is_bundle_slot_free(b: Bundle, instr: Instruction):
    match get_bundle_unit_type_of_instr(instr):
        case BundleUnitType.alu:
            if b.alu1 == None or b.alu2 == None:
                return True
            else:
                return False
        case BundleUnitType.mul:
            if b.mult == None:
                return True
            else:
                return False
        case BundleUnitType.mem:
            if b.mem == None:
                return True
            else:
                return False
        case BundleUnitType.branch:
            if b.branch == None:
                return True
            else:
                return False
        case BundleUnitType.anything:
            return True
        case _:
            print(
                f"[ERROR] Unhandled instruction type {get_bundle_unit_type_of_instr(instr)} in is_bundle_slot_free for schedule pip"
            )
            return False


def earliest_free_slot(schedule: list[Bundle], instr: Instruction):
    pc = len(schedule)
    unit_type = get_bundle_unit_type_of_instr(instr)
    for i, b in enumerate(schedule):
        match unit_type:
            case BundleUnitType.alu:
                if b.alu1 == None or b.alu2 == None:
                    pc = i
                    break
            case BundleUnitType.mul:
                if b.mult == None:
                    pc = i
                    break
            case BundleUnitType.mem:
                if b.mem == None:
                    pc = i
                    break
            case BundleUnitType.branch:
                if b.branch == None:
                    pc = i
                    break
            case _:
                print(
                    f"[ERROR] Unhandled instruction type {instr.instruction_type} in earliest_free_slot for schedule pip"
                )
                break
    return pc


# End schedule pip fonction #
# ==================================================================================== #

# ==================================================================================== #
# renaming pip function #


def rename_in_bb(
    bb: list[Bundle],
    old_name: int,
    new_name: int,
    instr_new_dest: dict,
    rename_dest=False,
    rename_operand=False,
):
    for bundle in bb:
        for instr in bundle:
            if instr != None:
                if rename_dest:
                    if instr.get_dest() == old_name:
                        instr.set_dest(new_name)
                        instr_new_dest[instr.get_id()] = new_name
                if rename_operand:
                    instr.set_src(old_name, new_name)
    return bb, instr_new_dest


def stage_of(bb1_start: int, II: int, instr_pc: int):
    return (instr_pc - bb1_start) // II


def is_inter_dep_operand(
    bb1: list[Bundle], i_id: int, reg: int, dep_table: DependencyTable
):
    reg_is_inter_dep_op = False
    reg_producer = None
    prod_bundle_addr = 0
    for idx, b in enumerate(bb1):
        for i in b:
            if i != None:
                dep_entry = dep_table.get_instruction_dependencies(i.get_id())
                if reg in dep_entry.interloop_dep.keys():
                    reg_is_inter_dep_op = min(dep_entry.interloop_dep[reg]) == i_id
                if reg == dep_entry.get_dest():
                    reg_producer = i
                    prod_bundle_addr = idx
    return reg_is_inter_dep_op, reg_producer, prod_bundle_addr


# End renaming pip fonction #
# ==================================================================================== #

if __name__ == "__main__":
    main()
