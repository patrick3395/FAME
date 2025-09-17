import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { ChangeEvent, MouseEvent } from "react"
import type {
  DataValidationRule,
  Dependency,
  EquationSheet,
  EquationsData,
  Formula,
  TextCell,
  TextData,
  TextSheet,
} from "./types"
import {
  makeCellReference,
  numberToColumnLabel,
  parseCellReference,
} from "./utils/cellAddress"
import { expandRangeWithSheet } from "./utils/rangeUtils"
import "./App.css"

const MAX_RENDER_ROWS = 200
const MAX_RENDER_COLUMNS = 40
const MIN_ROWS = 30
const MIN_COLUMNS = 15

const dataBaseUrl = (import.meta.env.BASE_URL || "") + "data"
const TEXT_URL = dataBaseUrl + "/FAME_TEXT.json"
const EQUATIONS_URL = dataBaseUrl + "/FAME_EQUATIONS.json"

const DEV_BUILD_VERSION = "Version 3"


type LoadState<T> = {
  status: "idle" | "loading" | "ready" | "error"
  data: T | null
  error?: string
}

type SheetCellRecord = {
  reference: string
  text?: TextCell
  formula?: Formula
}

type SheetSummary = {
  name: string
  cellMap: Map<string, SheetCellRecord>
  maxRow: number
  maxColumn: number
  firstCell: string
}

type GridMetrics = {
  totalRows: number
  totalColumns: number
  rowCount: number
  columnCount: number
  rowsTruncated: boolean
  columnsTruncated: boolean
  rowIndices: number[]
  columnHeaders: { index: number; label: string }[]
}

type ValidationLookup = Map<string, DataValidationRule>
type DependencyMap = Map<string, Set<string>>

const makeKey = (sheet: string, cell: string) => `${sheet}!${cell.toUpperCase()}`

function App() {
  console.debug('[banner-version]', DEV_BUILD_VERSION)
  const [textsState, setTextsState] = useState<LoadState<TextData>>({
    status: "idle",
    data: null,
  })
  const [equationsState, setEquationsState] =
    useState<LoadState<EquationsData>>({
      status: "idle",
      data: null,
    })
  const [selectedSheetName, setSelectedSheetName] = useState<string | null>(null)
  const [activeCell, setActiveCell] = useState<string>("A1")
  const [cellValues, setCellValues] = useState<Record<string, string>>({})
  const [filterQuery, setFilterQuery] = useState("")

  useEffect(() => {
    const loadTexts = async () => {
      setTextsState({ status: "loading", data: null })
      try {
        const response = await fetch(TEXT_URL)
        if (!response.ok) {
          throw new Error("Failed to load text data (" + response.status + ")")
        }
        const payload: TextData = await response.json()
        setTextsState({ status: "ready", data: payload })
      } catch (error) {
        setTextsState({
          status: "error",
          data: null,
          error: error instanceof Error ? error.message : "Unknown error",
        })
      }
    }

    const loadEquations = async () => {
      setEquationsState({ status: "loading", data: null })
      try {
        const response = await fetch(EQUATIONS_URL)
        if (!response.ok) {
          throw new Error(
            "Failed to load equation data (" + response.status + ")",
          )
        }
        const payload: EquationsData = await response.json()
        setEquationsState({ status: "ready", data: payload })
      } catch (error) {
        setEquationsState({
          status: "error",
          data: null,
          error: error instanceof Error ? error.message : "Unknown error",
        })
      }
    }

    loadTexts()
    loadEquations()
  }, [])

  const textSheets = useMemo(() => {
    if (textsState.status !== "ready" || !textsState.data) {
      return [] as TextSheet[]
    }
    return textsState.data.workbook.sheets
  }, [textsState])

  const equationSheets = useMemo(() => {
    if (equationsState.status !== "ready" || !equationsState.data) {
      return [] as EquationSheet[]
    }
    return equationsState.data.workbook.sheets
  }, [equationsState])

  const textSheetMap = useMemo(() => {
    const map = new Map<string, TextSheet>()
    textSheets.forEach((sheet) => {
      map.set(sheet.name, sheet)
    })
    return map
  }, [textSheets])

  const equationSheetMap = useMemo(() => {
    const map = new Map<string, EquationSheet>()
    equationSheets.forEach((sheet) => {
      map.set(sheet.name, sheet)
    })
    return map
  }, [equationSheets])

  const textCellMap = useMemo(() => {
    const map = new Map<string, TextCell>()
    textSheets.forEach((sheet) => {
      sheet.texts.forEach((cell) => {
        map.set(makeKey(sheet.name, cell.cell), cell)
      })
    })
    return map
  }, [textSheets])

  const formulaMap = useMemo(() => {
    const map = new Map<string, Formula>()
    equationSheets.forEach((sheet) => {
      sheet.formulas?.forEach((formula) => {
        map.set(makeKey(sheet.name, formula.cell), formula)
      })
    })
    return map
  }, [equationSheets])

  const validationLookup = useMemo<ValidationLookup>(() => {
    const map: ValidationLookup = new Map()
    if (equationsState.status !== "ready" || !equationsState.data) {
      return map
    }
    equationsState.data.workbook.sheets.forEach((sheet) => {
      sheet.data_validation?.forEach((rule) => {
        rule.range
          .split(",")
          .map((part) => part.trim())
          .filter(Boolean)
          .forEach((part) => {
            const { cells } = expandRangeWithSheet(part, sheet.name)
            cells.forEach((reference) => {
              map.set(makeKey(sheet.name, reference), rule)
            })
          })
      })
    })
    return map
  }, [equationsState])

  const dependencyMaps = useMemo(() => {
    const precedents: DependencyMap = new Map()
    const dependents: DependencyMap = new Map()
    if (equationsState.status !== "ready" || !equationsState.data) {
      return { precedents, dependents }
    }
    const deps = equationsState.data.workbook.dependencies ?? []
    deps.forEach((dependency: Dependency) => {
      const from = dependency.from.toUpperCase()
      const to = dependency.to.toUpperCase()
      if (!precedents.has(from)) {
        precedents.set(from, new Set())
      }
      precedents.get(from)?.add(to)
      if (!dependents.has(to)) {
        dependents.set(to, new Set())
      }
      dependents.get(to)?.add(from)
    })
    return { precedents, dependents }
  }, [equationsState])

  const getValidationOptionsForCell = useCallback(
    (sheetName: string, reference: string) => {
      const key = makeKey(sheetName, reference.toUpperCase())
      const rule = validationLookup.get(key)
      if (!rule || rule.type !== "list") {
        return { options: [] as string[], allowBlank: rule?.allow_blank ?? false }
      }
      const formula = rule.formula1 ?? ""
      let options: string[] = []
      if (formula && formula.startsWith("\"") && formula.endsWith("\"")) {
        options = formula
          .slice(1, -1)
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean)
      } else if (formula) {
        const expanded = expandRangeWithSheet(formula, sheetName)
        options = expanded.cells
          .map((ref) => getCellValue(expanded.sheet, ref))
          .filter((value) => value !== "")
        options = Array.from(new Set(options))
      }
      return { options, allowBlank: rule.allow_blank ?? false }
    },
    [validationLookup, cellValues]
  )

  useEffect(() => {
    if (textsState.status !== "ready" || !textsState.data) {
      return
    }
    const initialEntries: Record<string, string> = {}
    textsState.data.workbook.sheets.forEach((sheet) => {
      sheet.texts.forEach((cell) => {
        const key = makeKey(sheet.name, cell.cell)
        const value = cell.computed_text ?? cell.original_text ?? ""
        initialEntries[key] = value ?? ""
      })
    })
    setCellValues(initialEntries)
  }, [textsState])

  useEffect(() => {
    if (!selectedSheetName && textSheets.length > 0) {
      setSelectedSheetName(textSheets[0].name)
    }
  }, [selectedSheetName, textSheets])

  const sheetSummary = useMemo<SheetSummary | null>(() => {
    if (!selectedSheetName) {
      return null
    }
    const map = new Map<string, SheetCellRecord>()
    let maxRow = 0
    let maxColumn = 0
    let firstCell: string | null = null

    const considerReference = (reference: string) => {
      const coords = parseCellReference(reference)
      if (!coords) {
        return
      }
      maxRow = Math.max(maxRow, coords.row)
      maxColumn = Math.max(maxColumn, coords.column)
      if (!firstCell) {
        firstCell = reference
        return
      }
      const current = parseCellReference(firstCell)
      if (!current) {
        firstCell = reference
        return
      }
      if (coords.row < current.row) {
        firstCell = reference
      } else if (coords.row === current.row && coords.column < current.column) {
        firstCell = reference
      }
    }

    const textSheet = textSheetMap.get(selectedSheetName)
    textSheet?.texts.forEach((cell) => {
      const reference = cell.cell.toUpperCase()
      considerReference(reference)
      const existing = map.get(reference)
      if (existing) {
        existing.text = cell
      } else {
        map.set(reference, {
          reference,
          text: cell,
        })
      }
    })

    const equationSheet = equationSheetMap.get(selectedSheetName)
    equationSheet?.formulas?.forEach((formula) => {
      const reference = formula.cell.toUpperCase()
      considerReference(reference)
      const existing = map.get(reference)
      if (existing) {
        existing.formula = formula
      } else {
        map.set(reference, {
          reference,
          formula,
        })
      }
    })

    if (equationSheet?.dimensions) {
      maxRow = Math.max(maxRow, equationSheet.dimensions.rows ?? 0)
      maxColumn = Math.max(maxColumn, equationSheet.dimensions.cols ?? 0)
    }

    if (!firstCell) {
      firstCell = "A1"
    }

    return {
      name: selectedSheetName,
      cellMap: map,
      maxRow: Math.max(maxRow, MIN_ROWS),
      maxColumn: Math.max(maxColumn, MIN_COLUMNS),
      firstCell,
    }
  }, [selectedSheetName, textSheetMap, equationSheetMap])

  useEffect(() => {
    if (!sheetSummary) {
      return
    }
    setActiveCell(sheetSummary.firstCell)
  }, [sheetSummary?.firstCell])

  const gridMetrics = useMemo<GridMetrics>(() => {
    if (!sheetSummary) {
      return {
        totalRows: MIN_ROWS,
        totalColumns: MIN_COLUMNS,
        rowCount: MIN_ROWS,
        columnCount: MIN_COLUMNS,
        rowsTruncated: false,
        columnsTruncated: false,
        rowIndices: Array.from({ length: MIN_ROWS }, (_, index) => index + 1),
        columnHeaders: Array.from({ length: MIN_COLUMNS }, (_, index) => ({
          index: index + 1,
          label: numberToColumnLabel(index + 1),
        })),
      }
    }
    const totalRows = Math.max(sheetSummary.maxRow, MIN_ROWS)
    const totalColumns = Math.max(sheetSummary.maxColumn, MIN_COLUMNS)
    const rowCount = Math.min(totalRows, MAX_RENDER_ROWS)
    const columnCount = Math.min(totalColumns, MAX_RENDER_COLUMNS)
    const rowsTruncated = totalRows > rowCount
    const columnsTruncated = totalColumns > columnCount
    const rowIndices = Array.from({ length: rowCount }, (_, index) => index + 1)
    const columnHeaders = Array.from({ length: columnCount }, (_, index) => ({
      index: index + 1,
      label: numberToColumnLabel(index + 1),
    }))
    return {
      totalRows,
      totalColumns,
      rowCount,
      columnCount,
      rowsTruncated,
      columnsTruncated,
      rowIndices,
      columnHeaders,
    }
  }, [sheetSummary])

  const isLoading =
    textsState.status === "loading" || equationsState.status === "loading"

  const normalizedFilter = filterQuery.trim().toLowerCase()

  const matchesFilter = (sheet: string, reference: string) => {
    if (!normalizedFilter) {
      return true
    }
    const key = makeKey(sheet, reference)
    const textCell = textCellMap.get(key)
    const formula = formulaMap.get(key)
    const value = getCellValue(sheet, reference)
    const haystack = [
      reference,
      textCell?.original_text ?? "",
      textCell?.computed_text ?? "",
      formula?.a1 ?? "",
      value,
    ]
      .join(" ")
      .toLowerCase()
    return haystack.includes(normalizedFilter)
  }

  const selectedCellKey =
    selectedSheetName && activeCell
      ? makeKey(selectedSheetName, activeCell)
      : null
  const selectedTextCell = selectedCellKey
    ? textCellMap.get(selectedCellKey)
    : undefined
  const selectedFormula = selectedCellKey
    ? formulaMap.get(selectedCellKey)
    : undefined
  const selectedValidation = selectedCellKey
    ? validationLookup.get(selectedCellKey)
    : undefined

  const selectedDependencies = selectedFormula?.depends_on ?? []
  const selectedDependents = selectedCellKey
    ? Array.from(dependencyMaps.dependents.get(selectedCellKey) ?? [])
    : []

  function getCellValue(sheet: string, reference: string): string {
    const key = makeKey(sheet, reference)
    if (Object.prototype.hasOwnProperty.call(cellValues, key)) {
      return cellValues[key] ?? ""
    }
    const textCell = textCellMap.get(key)
    if (textCell?.computed_text) {
      return textCell.computed_text
    }
    if (textCell?.original_text) {
      return textCell.original_text
    }
    return ""
  }

  function getOriginalCellValue(sheet: string, reference: string): string {
    const key = makeKey(sheet, reference)
    const textCell = textCellMap.get(key)
    if (textCell?.original_text) {
      return textCell.original_text
    }
    if (textCell?.computed_text) {
      return textCell.computed_text
    }
    return ""
  }

  const validationMeta = useMemo(() => {
    if (!selectedSheetName || !selectedCellKey) {
      return { options: [] as string[], allowBlank: false }
    }
    return getValidationOptionsForCell(selectedSheetName, activeCell)
  }, [selectedSheetName, selectedCellKey, activeCell, getValidationOptionsForCell])

  const handleCellValueChange = (sheet: string, reference: string, value: string) => {
    setCellValues((previous) => ({
      ...previous,
      [makeKey(sheet, reference)]: value,
    }))
  }

  const handleResetCellValue = (sheet: string, reference: string) => {
    const key = makeKey(sheet, reference)
    const original = getOriginalCellValue(sheet, reference)
    setCellValues((previous) => ({
      ...previous,
      [key]: original,
    }))
  }

  const handleNavigateToReference = (reference: string) => {
    const [sheet, cell] = reference.split("!")
    if (sheet && cell) {
      setSelectedSheetName(sheet)
      setActiveCell(cell.toUpperCase())
    } else if (selectedSheetName) {
      setActiveCell(reference.toUpperCase())
    }
  }

  const renderLoadError = (label: string, state: LoadState<unknown>) => {
    if (state.status !== "error") {
      return null
    }
    return <p className="status error">Unable to load {label}: {state.error}</p>
  }

  return (
    <div className="app-root">
      <header className="app-banner" role="banner">
        <span className="app-version">{DEV_BUILD_VERSION}</span>
        <span className="app-env">Development Preview</span>
      </header>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <h1>FAME Workbook</h1>
            <p className="file-name">{textsState.data?.workbook.file_name ?? ""}</p>
          </div>
          {renderLoadError("text", textsState)}
          {renderLoadError("equations", equationsState)}
          {isLoading && <p className="status">Loading workbook...</p>}
          <nav className="sheet-nav" aria-label="Workbook sheets">
            {textSheets.map((sheet) => {
              const isSelected = sheet.name === selectedSheetName
              return (
                <button
                  key={sheet.name}
                  type="button"
                  className={isSelected ? "sheet is-selected" : "sheet"}
                  onClick={() => setSelectedSheetName(sheet.name)}
                >
                  <span className="sheet-name">{sheet.name}</span>
                  <span className="sheet-meta">{sheet.used_range ?? ""}</span>
                </button>
              )
            })}
          </nav>
        </aside>
        <main className="sheet-view">
          {!sheetSummary ? (
            <div className="empty-state">
              <h2>Select a sheet</h2>
              <p>Choose any worksheet from the left to inspect its data.</p>
            </div>
          ) : (
            <>
              <header className="sheet-toolbar">
                <div>
                  <h2>{sheetSummary.name}</h2>
                  {sheetSummary.name === "SETUP" ? (
                    <p className="muted">Workbook setup overview</p>
                  ) : (
                    <p className="muted">
                      Grid size: {gridMetrics.totalColumns.toLocaleString()} ×
                      {gridMetrics.totalRows.toLocaleString()}
                    </p>
                  )}
                </div>
                <div className="toolbar-actions">
                  {sheetSummary.name === "SETUP" ? (
                    <div className="setup-actions">
                      <button type="button" className="setup-action" disabled>
                        Refresh Setup
                      </button>
                      <button type="button" className="setup-action" disabled>
                        Export PDF
                      </button>
                    </div>
                  ) : (
                    <>
                      <label className="search">
                        <span>Find</span>
                        <input
                          value={filterQuery}
                          onChange={(event) => setFilterQuery(event.target.value)}
                          placeholder="Search text, cell, or formula"
                        />
                      </label>
                      {(gridMetrics.rowsTruncated || gridMetrics.columnsTruncated) && (
                        <span className="status muted">
                          Displaying first {gridMetrics.columnCount.toLocaleString()} columns ×
                          {gridMetrics.rowCount.toLocaleString()} rows
                        </span>
                      )}
                    </>
                  )}
                </div>
              </header>
              {sheetSummary.name === "SETUP" ? (
                <SetupSheet
                  getValue={(cell) => getCellValue(sheetSummary.name, cell)}
                  onChange={(cell, value) =>
                    handleCellValueChange(sheetSummary.name, cell, value)
                  }
                  onSelect={(cell) => setActiveCell(cell.toUpperCase())}
                  getValidation={(cell) =>
                    getValidationOptionsForCell(sheetSummary.name, cell)
                  }
                />
              ) : sheetSummary.name === "FP1" ? (
                <FloorPlanSheet onSelectCell={(ref) => setActiveCell(ref)} />
              ) : (
                <div className="grid-wrapper" role="region" aria-label="Sheet grid">
                  <table className="grid" role="grid">
                    <thead>
                      <tr>
                        <th className="corner" aria-hidden="true">
                          {sheetSummary.name}
                        </th>
                        {gridMetrics.columnHeaders.map((column) => (
                          <th
                            key={column.index}
                            scope="col"
                            className={
                              parseCellReference(activeCell)?.column === column.index
                                ? "column-heading is-active"
                                : "column-heading"
                            }
                          >
                            {column.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {gridMetrics.rowIndices.map((rowNumber) => {
                        const isActiveRow = parseCellReference(activeCell)?.row === rowNumber
                        return (
                          <tr key={rowNumber}>
                            <th
                              scope="row"
                              className={isActiveRow ? "row-heading is-active" : "row-heading"}
                            >
                              {rowNumber}
                            </th>
                            {gridMetrics.columnHeaders.map((column) => {
                              const reference = makeCellReference(rowNumber, column.index)
                              const cellKey = makeKey(sheetSummary.name, reference)
                              const cellRecord = sheetSummary.cellMap.get(reference)
                              const displayValue = getCellValue(
                                sheetSummary.name,
                                reference,
                              )
                              const isActiveCell = reference === activeCell
                              const withinFilter = matchesFilter(
                                sheetSummary.name,
                                reference,
                              )
                              const hasFormula = Boolean(cellRecord?.formula)
                              const classNames = ["cell"]
                              if (isActiveCell) {
                                classNames.push("is-active")
                              }
                              if (hasFormula) {
                                classNames.push("has-formula")
                              }
                              if (!displayValue) {
                                classNames.push("is-empty")
                              }
                              if (!withinFilter) {
                                classNames.push("is-muted")
                              }
                              return (
                                <td
                                  key={cellKey}
                                  role="gridcell"
                                  className={classNames.join(" ")}
                                  onClick={() => setActiveCell(reference)}
                                  title={cellRecord?.formula?.a1 ?? ""}
                                >
                                  <span className="cell-content">{displayValue}</span>
                                  {hasFormula ? <span className="formula-marker" aria-hidden="true" /> : null}
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </main>
        <aside className="inspector" aria-label="Cell inspector">
          {!sheetSummary || !selectedCellKey ? (
            <div className="inspector-empty">
              <h3>Select a cell</h3>
              <p>Click any cell in the grid to view its details.</p>
            </div>
          ) : (
            <div className="inspector-panel">
              <header className="inspector-header">
                <h3>{selectedCellKey}</h3>
                <p className="muted">
                  {selectedTextCell?.original_text ?? "Unnamed cell"}
                </p>
              </header>
              <section className="inspector-section">
                <h4>Value</h4>
                {selectedFormula ? (
                  <div className="readonly-field">
                    <label>Calculated output</label>
                    <input value={getCellValue(sheetSummary.name, activeCell)} readOnly />
                    <p className="muted">Formula-driven; edit precedent cells to modify.</p>
                  </div>
                ) : (
                  <div className="edit-field">
                    <label>Cell input</label>
                    {validationMeta.options.length > 0 ? (
                      <select
                        value={getCellValue(sheetSummary.name, activeCell)}
                        onChange={(event) =>
                          handleCellValueChange(
                            sheetSummary.name,
                            activeCell,
                            event.target.value,
                          )
                        }
                      >
                        {validationMeta.allowBlank && <option value="">(blank)</option>}
                        {validationMeta.options.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        value={getCellValue(sheetSummary.name, activeCell)}
                        onChange={(event) =>
                          handleCellValueChange(
                            sheetSummary.name,
                            activeCell,
                            event.target.value,
                          )
                        }
                        placeholder="Enter value"
                      />
                    )}
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() =>
                        handleResetCellValue(sheetSummary.name, activeCell)
                      }
                    >
                      Reset to original
                    </button>
                  </div>
                )}
                {selectedValidation ? (
                  <div className="meta-block">
                    <h5>Validation</h5>
                    <p className="muted">
                      Type: {selectedValidation.type}
                      {selectedValidation.allow_blank ? " • blanks allowed" : ""}
                    </p>
                    {selectedValidation.formula1 && (
                      <p className="muted code-inline">
                        Source: {selectedValidation.formula1}
                      </p>
                    )}
                  </div>
                ) : null}
              </section>
              {selectedFormula ? (
                <section className="inspector-section">
                  <h4>Formula</h4>
                  <code className="formula">{selectedFormula.a1}</code>
                  {selectedDependencies.length > 0 ? (
                    <div className="meta-block">
                      <h5>References</h5>
                      <div className="chip-row">
                        {selectedDependencies.map((dependency) => (
                          <button
                            key={dependency}
                            type="button"
                            className="chip"
                            onClick={() => handleNavigateToReference(dependency)}
                          >
                            {dependency}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p className="muted">No direct precedents recorded.</p>
                  )}
                </section>
              ) : null}
              {selectedDependents.length > 0 ? (
                <section className="inspector-section">
                  <h4>Dependent cells</h4>
                  <div className="chip-column">
                    {selectedDependents.map((dependent) => (
                      <button
                        key={dependent}
                        type="button"
                        className="chip"
                        onClick={() => handleNavigateToReference(dependent)}
                      >
                        {dependent}
                      </button>
                    ))}
                  </div>
                </section>
              ) : null}
              {selectedTextCell?.computed_text &&
              selectedTextCell.computed_text !== selectedTextCell.original_text ? (
                <section className="inspector-section">
                  <h4>Original vs computed</h4>
                  <p className="muted">
                    Original: {selectedTextCell.original_text ?? "(blank)"}
                  </p>
                  <p className="muted">
                    Computed snapshot: {selectedTextCell.computed_text}
                  </p>
                </section>
              ) : null}
            </div>
          )}
        </aside>
      </div>
    </div>
  )



function SetupSheet({
  getValue,
  onChange,
  onSelect,
  getValidation,
}: {
  getValue: (cell: string) => string
  onChange: (cell: string, value: string) => void
  onSelect: (cell: string) => void
  getValidation: (cell: string) => { options: string[]; allowBlank: boolean }
}) {
  const type = getValue("C2")
  const version = getValue("F3")
  const engCode = getValue("G2")
  const stage = getValue("I3")
  const street = getValue("C5")
  const city = getValue("C6")
  const state = getValue("E6")
  const zip = getValue("G6")
  const badgeLabel = getValue("E8")
  const badgeDate = getValue("G8")
  const badgeTitle = getValue("E9")
  const note = getValue("B14")

  const typeValidation = getValidation("C2")
  const versionValidation = getValidation("F3")
  const cityValidation = getValidation("C6")
  const stateValidation = getValidation("E6")
  const zipValidation = getValidation("G6")

  const renderControl = (
    value: string,
    cell: string,
    validation: { options: string[]; allowBlank: boolean },
  ) => {
    if (validation.options.length === 0) {
      return (
        <input
          value={value}
          onFocus={() => onSelect(cell)}
          onChange={(event) => onChange(cell, event.target.value)}
        />
      )
    }
    return (
      <select
        value={value}
        onFocus={() => onSelect(cell)}
        onChange={(event) => onChange(cell, event.target.value)}
      >
        {validation.allowBlank && <option value="">(blank)</option>}
        {validation.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    )
  }

  const formatDate = (value: string) => {
    if (!value) {
      return ""
    }
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.getTime())) {
      const month = parsed.getMonth() + 1
      const day = parsed.getDate()
      const year = parsed.getFullYear()
      return month + '/' + day + '/' + year
    }
    return value
  }

  return (
    <div className="setup-sheet">
      <div className="setup-card">
        <div className="setup-row setup-row-top">
          <div className="setup-field setup-field-type">
            <label onClick={() => onSelect("B2")}>Type</label>
            {renderControl(type, "C2", typeValidation)}
          </div>
          <div className="setup-field setup-field-version">
            <label onClick={() => onSelect("F2")}>Version</label>
            {renderControl(version, "F3", versionValidation)}
          </div>
          <button type="button" className="setup-code" onClick={() => onSelect("G2")}>
            {engCode || ""}
          </button>
          <button type="button" className="setup-stage" onClick={() => onSelect("I3")}>
            <span className="setup-stage-label">Stage</span>
            <span className="setup-stage-value">{stage || ""}</span>
          </button>
        </div>
        <div className="setup-row setup-row-address">
          <div className="setup-field setup-field-street">
            <label onClick={() => onSelect("B5")}>Street Address</label>
            {renderControl(street, "C5", { options: [], allowBlank: true })}
          </div>
        </div>
        <div className="setup-row setup-row-address-line">
          <div className="setup-field">
            <label onClick={() => onSelect("C6")}>City</label>
            {renderControl(city, "C6", cityValidation)}
          </div>
          <div className="setup-field setup-field-state">
            <label onClick={() => onSelect("E6")}>State</label>
            {renderControl(state, "E6", stateValidation)}
          </div>
          <div className="setup-field setup-field-zip">
            <label onClick={() => onSelect("G6")}>ZIP</label>
            {renderControl(zip, "G6", zipValidation)}
          </div>
        </div>
        <button type="button" className="setup-hero" onClick={() => onSelect("E9")}>
          <div className="setup-hero-header">
            <span
              className="setup-hero-label"
              onClick={(event) => {
                event.stopPropagation()
                onSelect("E8")
              }}
            >
              {badgeLabel}
            </span>
            <span
              className="setup-hero-date"
              onClick={(event) => {
                event.stopPropagation()
                onSelect("G8")
              }}
            >
              {formatDate(badgeDate)}
            </span>
          </div>
          <div className="setup-hero-title">{badgeTitle || ""}</div>
        </button>
        <div className="setup-actions">
          <button type="button" className="setup-action" disabled>
            Refresh Setup
          </button>
          <button type="button" className="setup-action" disabled>
            Export PDF
          </button>
        </div>
        <button type="button" className="setup-note" onClick={() => onSelect("B14")}>
          {note}
        </button>
      </div>
    </div>
  )
}



function FloorPlanSheet({ onSelectCell }: { onSelectCell: (ref: string) => void }) {
  const ticks = useMemo(() => Array.from({ length: 51 }, (_, index) => index * 2), [])
  const cellCount = ticks.length - 1
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [overlaySrc, setOverlaySrc] = useState<string | null>(null)
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: string }>>([])

  useEffect(() => () => {
    if (overlaySrc) {
      URL.revokeObjectURL(overlaySrc)
    }
  }, [overlaySrc])

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleOverlayChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    const url = URL.createObjectURL(file)
    setOverlaySrc((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous)
      }
      return url
    })
  }

  const handleCellClick = (event: MouseEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const xRatio = Math.max(Math.min((event.clientX - bounds.left) / bounds.width, 1), 0)
    const yRatio = Math.max(Math.min((event.clientY - bounds.top) / bounds.height, 1), 0)
    const columnIndex = Math.min(Math.floor(xRatio * cellCount), cellCount - 1)
    const rowIndex = Math.min(Math.floor(yRatio * cellCount), cellCount - 1)
    const label = window.prompt('Measurement label', '')
    if (!label) {
      return
    }
    const colLabel = ticks[columnIndex]
    const rowLabel = ticks[rowIndex]
    onSelectCell('FP1 (' + colLabel + 'ft, ' + rowLabel + 'ft)')
    setPoints((previous) => [
      ...previous,
      {
        x: (columnIndex + 0.5) / cellCount,
        y: (rowIndex + 0.5) / cellCount,
        label: label.trim(),
      },
    ])
  }

  return (
    <div className="fp-shell">
      <div className="fp-toolbar">
        <button type="button" className="fp-button" onClick={handleUploadClick}>
          Upload Overlay
        </button>
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleOverlayChange}
        />
        {overlaySrc ? <span className="fp-hint">Overlay applied</span> : <span className="fp-hint">No overlay</span>}
      </div>
      <div className="fp-grid">
        <div className="fp-top-labels">
          <div className="fp-corner" />
          {ticks.map((tick) => (
            <div key={tick} className="fp-col-label">
              {tick}
            </div>
          ))}
        </div>
        <div className="fp-body">
          <div className="fp-left-labels">
            {ticks.map((tick) => (
              <div key={tick} className="fp-row-label">
                {tick}
              </div>
            ))}
          </div>
          <div className="fp-grid-area" onClick={handleCellClick}>
            {overlaySrc ? <img src={overlaySrc} alt="Floor plan overlay" className="fp-overlay" /> : null}
            <div className="fp-grid-lines" />
            {points.map((point, index) => (
              <div
                key={point.label + '-' + index}
                className="fp-point"
                style={{ left: point.x * 100 + '%', top: point.y * 100 + '%' }}
              >
                <span className="fp-point-dot" />
                <span className="fp-point-label">{point.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}


export default App
