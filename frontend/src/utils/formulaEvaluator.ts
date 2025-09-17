import type { Formula, TextCell } from "../types"

interface EvaluationContext {
  formulas: Map<string, Formula>
  textCells: Map<string, TextCell>
  getCellValue: (sheet: string, cell: string) => string
}

export function createFormulaEvaluator(
  formulas: Map<string, Formula>,
  textCells: Map<string, TextCell>,
  getCellValue: (sheet: string, cell: string) => string
): (sheet: string, cell: string) => string {
  const cache = new Map<string, string>()
  const evaluating = new Set<string>()

  function evaluateFormula(sheet: string, cell: string): string {
    const key = `${sheet}!${cell}`

    // Check cache first
    if (cache.has(key)) {
      return cache.get(key)!
    }

    // Check for circular reference
    if (evaluating.has(key)) {
      return "#CIRCULAR!"
    }

    // Check if there's a text cell with computed_text
    const textCell = textCells.get(key)
    if (textCell?.computed_text) {
      cache.set(key, textCell.computed_text)
      return textCell.computed_text
    }

    // Check if there's a formula
    const formula = formulas.get(key)
    if (!formula) {
      // No formula, check for original text
      if (textCell?.original_text) {
        cache.set(key, textCell.original_text)
        return textCell.original_text
      }
      return ""
    }

    // Mark as evaluating to detect circular references
    evaluating.add(key)

    try {
      // Simple formula evaluation for common patterns
      let result = evaluateSimpleFormula(formula.a1, sheet, {
        formulas,
        textCells,
        getCellValue: (s: string, c: string) => {
          // Recursively evaluate dependencies
          return evaluateFormula(s, c)
        }
      })

      cache.set(key, result)
      return result
    } finally {
      evaluating.delete(key)
    }
  }

  return evaluateFormula
}

function evaluateSimpleFormula(
  formula: string,
  currentSheet: string,
  context: EvaluationContext
): string {
  // Remove leading =
  let expr = formula.startsWith("=") ? formula.slice(1) : formula

  // Handle sheet references like 'FP0'!E10
  expr = expr.replace(/'([^']+)'!([A-Z]+\d+)/g, (match, sheet, cell) => {
    const value = context.getCellValue(sheet, cell)
    // If it's a number, return it; otherwise quote it
    return isNaN(Number(value)) ? `"${value}"` : value
  })

  // Handle same-sheet references
  expr = expr.replace(/\b([A-Z]+\d+)\b/g, (match, cell) => {
    const value = context.getCellValue(currentSheet, cell)
    return isNaN(Number(value)) ? `"${value}"` : value
  })

  // Handle string concatenation (&)
  expr = expr.replace(/&/g, "+")

  // Handle basic arithmetic
  try {
    // Simple evaluation for basic math and string concatenation
    if (/^[\d\s+\-*/()."]+$/.test(expr)) {
      // Replace quoted strings for concatenation
      expr = expr.replace(/"([^"]*)"/g, "'$1'")

      // Use Function constructor for safe evaluation
      const result = new Function(`return ${expr}`)()
      return String(result)
    }
  } catch (e) {
    // If evaluation fails, return the formula
    return formula
  }

  // For complex formulas, return the original formula text
  return formula
}

export function makeKey(sheet: string, cell: string): string {
  return `${sheet}!${cell.toUpperCase()}`
}