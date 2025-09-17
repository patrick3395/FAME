const COLUMN_BASE = 26
const CHAR_CODE_A = "A".charCodeAt(0)

export function columnLabelToNumber(label: string): number {
  const cleaned = label.trim().toUpperCase()
  if (!cleaned) {
    return NaN
  }
  let value = 0
  for (let index = 0; index < cleaned.length; index += 1) {
    const code = cleaned.charCodeAt(index)
    if (code < CHAR_CODE_A || code > CHAR_CODE_A + 25) {
      return NaN
    }
    value = value * COLUMN_BASE + (code - CHAR_CODE_A + 1)
  }
  return value
}

export function numberToColumnLabel(index: number): string {
  if (index < 1) {
    return ""
  }
  let value = index
  let label = ""
  while (value > 0) {
    const remainder = (value - 1) % COLUMN_BASE
    label = String.fromCharCode(CHAR_CODE_A + remainder) + label
    value = Math.floor((value - 1) / COLUMN_BASE)
  }
  return label
}

export function parseCellReference(reference: string): { row: number; column: number } | null {
  const match = /^([A-Z]+)(\d+)$/.exec(reference.trim().toUpperCase())
  if (!match) {
    return null
  }
  const [, columnPart, rowPart] = match
  const column = columnLabelToNumber(columnPart)
  const row = Number.parseInt(rowPart, 10)
  if (Number.isNaN(column) || Number.isNaN(row)) {
    return null
  }
  return { row, column }
}

export function makeCellReference(row: number, column: number): string {
  const columnLabel = numberToColumnLabel(column)
  return `${columnLabel}${row}`
}
