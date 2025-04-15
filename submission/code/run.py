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


def read_instruction(instruction):
    instr = instruction.split(" ")
    op_code = instr[0]
    dest_reg_arch_tag = int(instr[1][1:-1])  # remove x and ,
    reg_a_arch_tag = int(instr[2][1:-1])

    if op_code == "addi":
        is_b_imm = True
        reg_b_read = int(instr[3])
    else:
        is_b_imm = False
        reg_b_read = int(instr[3][1:])

    return op_code, dest_reg_arch_tag, reg_a_arch_tag, (is_b_imm, reg_b_read)


def main():

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    instructions = read_instructions(
        input_path
    )  # "op_code" dest_reg_arch_tag reg_a_arch_tag reg_b_arch_tag/imm

    active_list = ActiveList()
    free_list = FreeList()
    exception_handler = ExceptionUnit()
    pc = PC()
    decoded_instruction_register = DecodedInstructionRegister()
    integer_queue = IntegerQueue()
    busy_bit_table = BusyBitTable()
    physical_register_file = PhysicalRegisterFile()
    register_map_table = RegisterMapTable()
    alus = [ALU() for _ in range(4)]

    issued_instructions = []
    forwarding_paths = []

    logs = []
    last_commited_pc = 0

    while not exception_handler.exception_status() and (
        last_commited_pc < len(instructions) - 1
    ):

        # Logging TODO : redo the logging
        step = {}
        step["ActiveList"] = active_list.to_json()
        step["BusyBitTable"] = busy_bit_table.to_json()
        step["DecodedPCs"] = decoded_instruction_register.to_json()
        step["Exception"] = exception_handler.exception_status()
        step["ExceptionPC"] = exception_handler.exception_pc()
        step["FreeList"] = free_list.to_json()
        step["IntegerQueue"] = integer_queue.to_json()
        step["PC"] = pc.to_json()
        step["PhysicalRegisterFile"] = physical_register_file.to_json()
        step["RegisterMapTable"] = register_map_table.to_json()
        logs.append(step)

        # Forwarding path shortcut
        forwarding_paths = {}
        for i in range(4):
            res = alus[i].get_result()
            if res is not None:
                forwarding_paths[res.dest_reg] = res
                if not res.is_exception():
                    val = res.get_value_result()
                    integer_queue.pull_forwarding_path(res.dest_reg, val)
                    busy_bit_table.pull_forwarding_path(res.dest_reg)
                    physical_register_file.pull_forwarding_path(res.dest_reg, val)

        # Commit stage
        num_done = active_list.get_num_next_done_instructions()
        if num_done > 4:  # fanout
            num_done = 4

        num_exception = active_list.get_next_exception()
        if num_exception >= 0 and num_exception < 4:
            num_done = num_exception  # The instruction with the exception will be the last one in active list (position 0)
        else:
            num_exception = -1

        if num_done > free_list.get_free_space():
            num_done = free_list.get_free_space()

        committed_instructions = active_list.commit_instruction(num_done)

        for i in range(num_done):
            free_list.add_free_register(
                committed_instructions[i].old_physical_destination_register
            )

        if len(committed_instructions) > 0:
            last_commited_pc = committed_instructions[-1].pc

        # Issue stage
        i = 0
        instr = integer_queue.get_instruction_by_index(i)
        while len(issued_instructions) < 4 and instr is not None:
            if instr.is_ready():
                issued_instructions.append(instr)
                integer_queue.remove_instruction(instr.pc)
            i += 1
            instr = integer_queue.get_instruction_by_index(i)

        # Execute stages
        for i in range(4):
            if len(issued_instructions) > 0 and issued_instructions[0] is not None:
                alus[i].execute(issued_instructions.pop(0))

            res = alus[i].get_result()
            if res is not None:
                active_list.update_instruction(res.pc, res.is_exception())
                if not res.is_exception():
                    val = res.get_value_result()
                    integer_queue.update_instruction(res.dest_reg, val)
                    busy_bit_table.clear_busy(res.dest_reg)
                    physical_register_file.set_register(res.dest_reg, val)
                    if val >= 18446744073709101615:
                        print(f"res : {res}")

        # Rename & decode stage
        num_decodable_instructions = (
            4  # for simplicity, either all decodable instructions are renamed or none
        )

        if (
            active_list.get_free_space() >= num_decodable_instructions
            and integer_queue.get_free_space() >= num_decodable_instructions
            and free_list.get_num_free_registers() >= num_decodable_instructions
        ):
            decoded_instructions = decoded_instruction_register.pull_instructions(
                num_decodable_instructions
            )
            for decoded_pc in decoded_instructions:

                op_code, dest_reg_arch_tag, reg_a_arch_tag, (is_b_imm, reg_b_read) = (
                    read_instruction(instructions[decoded_pc])
                )

                # Get physical registers before the mapping is changed (in case of add x0 x0 ..., 0 is renamed but we need to use the old 0)
                reg_a = register_map_table.get_register(reg_a_arch_tag)
                reg_b = (
                    None if is_b_imm else register_map_table.get_register(reg_b_read)
                )

                dest_phys_reg = free_list.pull_free_register()
                old_phys_reg = register_map_table.set_register(
                    dest_reg_arch_tag, dest_phys_reg
                )

                # Get values of registers
                reg_a_value = (
                    None
                    if busy_bit_table.is_busy(reg_a)
                    else physical_register_file.get_register(reg_a)
                )

                reg_b_value = (
                    reg_b_read
                    if is_b_imm
                    else (
                        None
                        if busy_bit_table.is_busy(reg_b)
                        else physical_register_file.get_register(reg_b)
                    )
                )

                busy_bit_table.set_busy(
                    dest_phys_reg
                )  # Set busy after getting the old values

                # Add instruction to integer queue
                # Value is None if the register is not ready
                integer_queue.add_instruction(
                    dest_phys_reg,
                    reg_a,
                    reg_a_value,
                    reg_b,
                    reg_b_value,
                    op_code,
                    decoded_pc,
                )

                # Add instruction to active list
                active_list.add_instruction(
                    dest_reg_arch_tag,
                    old_phys_reg,
                    decoded_pc,
                )

        # Fetch stage
        num_instructions_to_pull = decoded_instruction_register.get_free_space()
        if num_instructions_to_pull > len(instructions) - pc.get_pc():
            num_instructions_to_pull = len(instructions) - pc.get_pc()

        for i in range(num_instructions_to_pull):
            decoded_instruction_register.add_instruction(pc.increment_and_get())

        # Go to exception mode
        if num_exception >= 0 and num_exception < 4:
            exception_handler.start_exception(active_list.get_next_instruction().pc)
            pc.set_pc(0x10000)
            decoded_instruction_register.clear()
            integer_queue.clear()

        # Clock pos edge
        active_list.next_step()
        free_list.next_step()
        pc.next_step()
        decoded_instruction_register.next_step()
        integer_queue.next_step()
        busy_bit_table.next_step()
        physical_register_file.next_step()
        register_map_table.next_step()

        for i in range(4):
            alus[i].next_step()

        forwarding_paths = {}

    if exception_handler.exception_status():
        while not active_list.is_empty():
            # TODO: Exception handling
            # Similar to MIPS R10000, in every cycle, pick (up to) 4 instructions in reverse program order from the Active list bottom,
            # and use it to recover the Register Map Table, the Free List, and the Busy Bit Table.

            step = {}
            step["ActiveList"] = active_list.to_json()
            step["BusyBitTable"] = busy_bit_table.to_json()
            step["DecodedPCs"] = decoded_instruction_register.to_json()
            step["Exception"] = exception_handler.exception_status()
            step["ExceptionPC"] = exception_handler.exception_pc()
            step["FreeList"] = free_list.to_json()
            step["IntegerQueue"] = integer_queue.to_json()
            step["PC"] = pc.to_json()
            step["PhysicalRegisterFile"] = physical_register_file.to_json()
            step["RegisterMapTable"] = register_map_table.to_json()
            logs.append(step)

            # pick up to 4 instructions in reverse order (already putted in the reverse order by exception_next_step)
            instructions = active_list.exception_next_step()
            for instr in instructions:
                """
                My version of recovery (probably wrong since busy bit table is recovered instead of cleared)
                old_reg = register_map_table.exception_set_register(
                    instr.architectural_destination_register,
                    instr.old_physical_destination_register,
                )
                free_list.exception_add_free_register(old_reg)
                busy_bit_table.exception_clear_busy(
                    instr.old_physical_destination_register
                )
                """

                # Professor's version of recovery
                old_reg = register_map_table.exception_set_register(
                    instr.architectural_destination_register,
                    instr.old_physical_destination_register,
                )
                free_list.add_free_register(old_reg)
                busy_bit_table.exception_clear_busy(old_reg)

            free_list.next_step()

            # last log before going out of exception mode
            if active_list.is_empty():
                step = {}
                step["ActiveList"] = active_list.to_json()
                step["BusyBitTable"] = busy_bit_table.to_json()
                step["DecodedPCs"] = decoded_instruction_register.to_json()
                step["Exception"] = exception_handler.exception_status()
                step["ExceptionPC"] = exception_handler.exception_pc()
                step["FreeList"] = free_list.to_json()
                step["IntegerQueue"] = integer_queue.to_json()
                step["PC"] = pc.to_json()
                step["PhysicalRegisterFile"] = physical_register_file.to_json()
                step["RegisterMapTable"] = register_map_table.to_json()
                logs.append(step)

                exception_handler.end_exception()

    last_step = {}
    last_step["ActiveList"] = active_list.to_json()
    last_step["BusyBitTable"] = busy_bit_table.to_json()
    last_step["DecodedPCs"] = decoded_instruction_register.to_json()
    last_step["Exception"] = exception_handler.exception_status()
    last_step["ExceptionPC"] = exception_handler.exception_pc()
    last_step["FreeList"] = free_list.to_json()
    last_step["IntegerQueue"] = integer_queue.to_json()
    last_step["PC"] = pc.to_json()
    last_step["PhysicalRegisterFile"] = physical_register_file.to_json()
    last_step["RegisterMapTable"] = register_map_table.to_json()
    logs.append(last_step)
    write_log(logs, output_path)


if __name__ == "__main__":
    main()
