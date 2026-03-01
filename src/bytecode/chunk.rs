use super::opcode::{Instruction, Opcode};
use crate::Value;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<Instruction>,
    pub constants: Vec<Value>,
    pub name: String,
}

impl Chunk {
    pub fn new(name: impl Into<String>) -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            name: name.into(),
        }
    }

    pub fn write(&mut self, instruction: Instruction) -> usize {
        let index = self.code.len();
        self.code.push(instruction);
        index
    }

    pub fn write_simple(&mut self, opcode: Opcode, line: usize) -> usize {
        self.write(Instruction::simple(opcode, line))
    }

    pub fn write_u16(&mut self, opcode: Opcode, operand: u16, line: usize) -> usize {
        self.write(Instruction::with_u16(opcode, operand, line))
    }

    pub fn write_u8(&mut self, opcode: Opcode, operand: u8, line: usize) -> usize {
        self.write(Instruction::with_u8(opcode, operand, line))
    }

    pub fn add_constant(&mut self, value: Value) -> u16 {
        for (i, c) in self.constants.iter().enumerate() {
            if values_equal(c, &value) {
                return i as u16;
            }
        }
        let index = self.constants.len();
        self.constants.push(value);
        index as u16
    }

    pub fn patch_jump(&mut self, index: usize, offset: i16) {
        let bytes = offset.to_le_bytes();
        self.code[index].operands = vec![bytes[0], bytes[1]];
    }

    pub fn current_offset(&self) -> usize {
        self.code.len()
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

impl fmt::Display for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "== {} ==", self.name)?;
        for (i, instr) in self.code.iter().enumerate() {
            write!(f, "{:04} ", i)?;

            if i > 0 && instr.line == self.code[i - 1].line {
                write!(f, "   | ")?;
            } else {
                write!(f, "{:4} ", instr.line)?;
            }

            write!(f, "{:16}", format!("{:?}", instr.opcode))?;

            match instr.opcode {
                Opcode::Const => {
                    let idx = instr.read_u16();
                    if let Some(val) = self.constants.get(idx as usize) {
                        write!(f, " {:4} '{}'", idx, val)?;
                    } else {
                        write!(f, " {:4}", idx)?;
                    }
                }
                Opcode::GetLocal | Opcode::SetLocal => {
                    write!(f, " {:4}", instr.read_u16())?;
                }
                Opcode::GetGlobal
                | Opcode::SetGlobal
                | Opcode::DefineGlobal
                | Opcode::GetField
                | Opcode::SetField => {
                    let idx = instr.read_u16();
                    if let Some(Value::String(s)) = self.constants.get(idx as usize) {
                        write!(f, " {:4} '{}'", idx, s)?;
                    } else {
                        write!(f, " {:4}", idx)?;
                    }
                }
                Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue => {
                    let offset = instr.read_i16();
                    let target = (i as i32 + 1 + offset as i32) as usize;
                    write!(f, " {:4} -> {}", offset, target)?;
                }
                Opcode::Loop => {
                    let offset = instr.read_u16();
                    let target = i + 1 - offset as usize;
                    write!(f, " {:4} -> {}", offset, target)?;
                }
                Opcode::Call => {
                    write!(f, " {:4}", instr.read_u8())?;
                }
                Opcode::MakeArray => {
                    write!(f, " {:4}", instr.read_u16())?;
                }
                Opcode::CloseScope => {
                    write!(f, " count: {}", instr.read_u16())?;
                }
                _ => {}
            }

            writeln!(f)?;
        }

        if !self.constants.is_empty() {
            writeln!(f, "\n-- Constants --")?;
            for (i, c) in self.constants.iter().enumerate() {
                writeln!(f, "{:4}: {}", i, c)?;
            }
        }

        Ok(())
    }
}
