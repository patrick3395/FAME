import { makeCellReference, parseCellReference } from "./cellAddress"

export interface ParsedRange {
  sheet?: string
  start: { row: number; column: number }
  end: { row: number; column: number }
}

export function parseRange(range: string): ParsedRange | null {
  if (!range) {
    return null
  }
  const raw = range.trim()
  if (!raw) {
    return null
  }
  let sheet: string | undefined
  let reference = raw
  if (reference.includes("!")) {
    const parts = reference.split("!")
    sheet = parts[0].replace(/^'/, "").replace(/'$/, "")
    reference = parts[1]
  }
  const cleaned = reference.replace(/$/g, "")
  const [startRef, endRef] = cleaned.split(":")
  const start = parseCellReference(startRef)
  const end = parseCellReference(endRef ?? startRef)
  if (!start || !end) {
    return null
  }
  return {
    sheet,
    start: {
      row: Math.min(start.row, end.row),
      column: Math.min(start.column, end.column),
    },
    end: {
      row: Math.max(start.row, end.row),
      column: Math.max(start.column, end.column),
    },
  }
}

export function expandRange(range: string): string[] {
  const parsed = parseRange(range)
  if (!parsed) {
    return []
  }
  const cells: string[] = []
  for (let row = parsed.start.row; row <= parsed.end.row; row += 1) {
    for (let column = parsed.start.column; column <= parsed.end.column; column += 1) {
      cells.push(makeCellReference(row, column))
    }
  }
  return cells
}

export function expandRangeWithSheet(range: string, defaultSheet: string): {
  sheet: string
  cells: string[]
} {
  const parsed = parseRange(range)
  if (!parsed) {
    return { sheet: defaultSheet, cells: [] }
  }
  const sheet = parsed.sheet ?? defaultSheet
  const cells: string[] = []
  for (let row = parsed.start.row; row <= parsed.end.row; row += 1) {
    for (let column = parsed.start.column; column <= parsed.end.column; column += 1) {
      cells.push(makeCellReference(row, column))
    }
  }
  return { sheet, cells }
}

export function isCellInRange(cell: string, range: string): boolean {
  const parsedRange = parseRange(range)
  const parsedCell = parseCellReference(cell)
  if (!parsedRange || !parsedCell) {
    return false
  }
  return (
    parsedCell.row >= parsedRange.start.row &&
    parsedCell.row <= parsedRange.end.row &&
    parsedCell.column >= parsedRange.start.column &&
    parsedCell.column <= parsedRange.end.column
  )
}
