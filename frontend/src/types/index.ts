export interface TextCell {
  cell: string
  original_text: string
  computed_text?: string
}

export interface TextSheet {
  name: string
  used_range?: string
  texts: TextCell[]
}

export interface TextWorkbook {
  file_name: string
  sheets: TextSheet[]
}

export interface TextData {
  workbook: TextWorkbook
}

export interface Formula {
  cell: string
  a1: string
  r1c1?: string | null
  depends_on?: string[]
  outputs_to?: string[]
}

export interface DataValidationRule {
  range: string
  type: string
  operator?: string | null
  formula1?: string | null
  formula2?: string | null
  allow_blank?: boolean
}

export interface EquationSheet {
  name: string
  index: number
  dimensions?: {
    rows?: number
    cols?: number
    used_range?: string
  }
  formulas?: Formula[]
  data_validation?: DataValidationRule[]
}

export interface Dependency {
  from: string
  to: string
}

export interface NamedRange {
  name: string
  ref: string[]
}

export interface EquationsWorkbook {
  file_name: string
  sheets: EquationSheet[]
  dependencies?: Dependency[]
  named_ranges?: NamedRange[]
}

export interface EquationsData {
  workbook: EquationsWorkbook
}

export interface CalculationResults {
  [sheetName: string]: unknown
}

export interface SheetCell {
  reference: string
  displayValue: string
  originalText?: string
  computedText?: string
  formula?: Formula
}
