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

const DEV_BUILD_VERSION = "Version 34"


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
type Fp1PreviewPayload = {
  boundary: Array<{ x: number; y: number }>
  points: Array<{ x: number; y: number; z: number; label?: string }>
  spacing: number
  unit: string
  floorplanImage?: string | null
}

type InitialGraphic = {
  name: string
  image: string
}

type InitialGraphicsState = {
  status: 'idle' | 'loading' | 'ready' | 'error'
  graphics: InitialGraphic[]
  message?: string
  error?: string
}

const GRAPHIC_LABELS: Record<string, string> = {
  heatmap: 'Heatmap Layer',
  repair_plan: 'Contour Layer',
  contours: 'Contour Layer',
  profiles: 'Profile Lines',
  '1 - Elevation Plot': 'Heatmap Layer',
  '2 - Contours Mesh': 'Contour Layer',
  '3 - All Profiles': 'Profile Lines',
}

const FAME_API_ENDPOINT =
  (import.meta.env.VITE_FAME_API_ENDPOINT as string | undefined) ??
  'https://fameuideployment-245923252465.us-south1.run.app/api/fame/run'

const FAME_API_ENDPOINT_DISPLAY = FAME_API_ENDPOINT.startsWith('http')
  ? FAME_API_ENDPOINT
  : 'local proxy (' + FAME_API_ENDPOINT + ')'

const makeKey = (sheet: string, cell: string) => `${sheet}!${cell.toUpperCase()}`

function App() {
  console.debug('[banner-version]', DEV_BUILD_VERSION)
  console.info('[config] Using FAME endpoint', FAME_API_ENDPOINT)
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
                <FloorPlanSheet
                  summary={sheetSummary}
                  getValue={(cell) => getCellValue(sheetSummary.name, cell)}
                  onSelectCell={(ref) => setActiveCell(ref)}
                />
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
}


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





function FloorPlanSheet({
  summary,
  getValue,
  onSelectCell,
}: {
  summary: SheetSummary
  getValue: (cell: string) => string
  onSelectCell: (ref: string) => void
}) {
  const rawSpacing = getValue('A1')
  const parsedSpacing = Number.parseFloat(rawSpacing)
  const spacing = Number.isFinite(parsedSpacing) && parsedSpacing > 0 ? parsedSpacing : 2
  const normalized = rawSpacing.toLowerCase()
  let unit = 'ft'
  if (normalized.includes('meter') || normalized.includes('metre')) {
    unit = 'm'
  } else if (normalized.includes('in')) {
    unit = 'in'
  }

  const columnCells = Math.max(summary.maxColumn - 1, 1)
  const rowCells = Math.max(summary.maxRow - 1, 1)
  const columnTicks = useMemo(
    () => Array.from({ length: columnCells + 1 }, (_, index) => index * spacing),
    [columnCells, spacing],
  )
  const rowTicks = useMemo(
    () => Array.from({ length: rowCells + 1 }, (_, index) => index * spacing),
    [rowCells, spacing],
  )
  const columnCellCount = Math.max(columnTicks.length - 1, 1)
  const rowCellCount = Math.max(rowTicks.length - 1, 1)
  const maxXDistance = columnTicks[columnCellCount] !== undefined ? columnTicks[columnCellCount] : spacing * columnCellCount
  const maxYDistance = rowTicks[rowCellCount] !== undefined ? rowTicks[rowCellCount] : spacing * rowCellCount

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [overlaySrc, setOverlaySrc] = useState<string | null>(null)
  const [overlayDataUrl, setOverlayDataUrl] = useState<string | null>(null)
  const [overlayScale, setOverlayScale] = useState(1)
  const [mode, setMode] = useState<'measure' | 'border'>('measure')
  const borderIdRef = useRef(1)
  const pointIdRef = useRef(1)
  const [borderPoints, setBorderPoints] = useState<
    Array<{ id: string; x: number; y: number; nx: number; ny: number }>
  >([])
  const [measurements, setMeasurements] = useState<
    Array<{ id: string; x: number; y: number; nx: number; ny: number; z: number; label: string }>
  >([])
  const [isBorderClosed, setIsBorderClosed] = useState(false)
  const [showPayload, setShowPayload] = useState(false)
  const [payloadWarning, setPayloadWarning] = useState<string | null>(null)
  const [initialGraphicsState, setInitialGraphicsState] = useState<InitialGraphicsState>({
    status: 'idle',
    graphics: [],
  })

  const unitDisplay = unit || 'ft'

  const formatTick = (value: number) => {
    const rounded = Number.parseFloat(value.toFixed(2))
    return unitDisplay ? rounded + ' ' + unitDisplay : String(rounded)
  }

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
      setOverlaySrc((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous)
        }
        return null
      })
      setOverlayDataUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setOverlaySrc((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous)
      }
      return url
    })
    setOverlayScale(1)
    const reader = new FileReader()
    reader.onloadend = () => {
      const result = reader.result
      if (typeof result === 'string') {
        setOverlayDataUrl(result)
      } else {
        setOverlayDataUrl(null)
      }
    }
    reader.readAsDataURL(file)
  }

  const handleCellClick = (event: MouseEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const xRatio = Math.max(Math.min((event.clientX - bounds.left) / bounds.width, 1), 0)
    const yRatio = Math.max(Math.min((event.clientY - bounds.top) / bounds.height, 1), 0)

    if (mode === 'border') {
      const id = 'B' + borderIdRef.current
      borderIdRef.current += 1
      const xFeet = xRatio * maxXDistance
      const yFeet = yRatio * maxYDistance
      setBorderPoints((previous) => [
        ...previous,
        { id, x: xFeet, y: yFeet, nx: xRatio, ny: yRatio },
      ])
      setIsBorderClosed(false)
      setShowPayload(false)
      setPayloadWarning(null)
      return
    }

    const columnIndex = Math.min(Math.floor(xRatio * columnCellCount), columnCellCount - 1)
    const rowIndex = Math.min(Math.floor(yRatio * rowCellCount), rowCellCount - 1)
    const valueInput = window.prompt('Enter elevation value for this point', '')
    if (valueInput === null) {
      return
    }
    const numericValue = Number.parseFloat(valueInput)
    if (!Number.isFinite(numericValue)) {
      window.alert('Please enter a numeric value for the elevation.')
      return
    }
    const defaultLabel = 'Point ' + pointIdRef.current
    const labelInput = window.prompt('Measurement label', defaultLabel) ?? defaultLabel
    const sanitizedLabel = labelInput.trim() || defaultLabel
    const cellStartX = columnTicks[columnIndex]
    const cellEndX = columnTicks[Math.min(columnIndex + 1, columnCellCount)]
    const cellStartY = rowTicks[rowIndex]
    const cellEndY = rowTicks[Math.min(rowIndex + 1, rowCellCount)]
    const pointXFeet = (cellStartX + cellEndX) / 2
    const pointYFeet = (cellStartY + cellEndY) / 2
    const normalizedX = columnCellCount > 0 ? (columnIndex + 0.5) / columnCellCount : 0
    const normalizedY = rowCellCount > 0 ? (rowIndex + 0.5) / rowCellCount : 0
    const formattedCol = formatTick(pointXFeet)
    const formattedRow = formatTick(pointYFeet)
    const selectionLabel = 'FP1 (' + formattedCol + ', ' + formattedRow + ')'
    onSelectCell(selectionLabel)
    const id = 'P' + pointIdRef.current
    pointIdRef.current += 1
    setMeasurements((previous) => [
      ...previous,
      {
        id,
        x: pointXFeet,
        y: pointYFeet,
        nx: normalizedX,
        ny: normalizedY,
        z: numericValue,
        label: sanitizedLabel,
      },
    ])
    setShowPayload(false)
    setPayloadWarning(null)
  }

  const handleToggleMode = () => {
    setMode((previous) => (previous === 'border' ? 'measure' : 'border'))
  }

  const handleCloseBorder = () => {
    if (borderPoints.length >= 3) {
      setIsBorderClosed(true)
      setShowPayload(false)
      setPayloadWarning(null)
    }
  }

  const handleUndoBorderPoint = () => {
    setBorderPoints((previous) => previous.slice(0, -1))
    setIsBorderClosed(false)
    setShowPayload(false)
    setPayloadWarning(null)
  }

  const handleClearBorder = () => {
    setBorderPoints([])
    setIsBorderClosed(false)
    setShowPayload(false)
    setPayloadWarning(null)
  }

  const handleClearMeasurements = () => {
    setMeasurements([])
    setShowPayload(false)
    setPayloadWarning(null)
  }

  const handleRemoveMeasurement = () => {
    setMeasurements((previous) => previous.slice(0, -1))
    setShowPayload(false)
    setPayloadWarning(null)
  }

  const borderSvgPoints = borderPoints
    .map((point) => (point.nx * 100).toFixed(2) + ',' + (point.ny * 100).toFixed(2))
    .join(' ')

  const overlayTransform = 'translate(-50%, -50%) scale(' + overlayScale + ')'

  const payload = useMemo<Fp1PreviewPayload | null>(() => {
    if (!isBorderClosed || borderPoints.length < 3 || measurements.length === 0) {
      return null
    }
    const boundary = borderPoints.map((point) => ({
      x: Number.parseFloat(point.x.toFixed(3)),
      y: Number.parseFloat(point.y.toFixed(3)),
    }))
    if (boundary.length >= 3) {
      const first = boundary[0]
      const last = boundary[boundary.length - 1]
      if (first.x != last.x || first.y != last.y) {
        boundary.push({ x: first.x, y: first.y })
      }
    }
    const points = measurements.map((point) => ({
      x: Number.parseFloat(point.x.toFixed(3)),
      y: Number.parseFloat(point.y.toFixed(3)),
      z: Number.parseFloat(point.z.toFixed(3)),
      label: point.label,
    }))
    return {
      boundary,
      points,
      spacing,
      unit: unitDisplay,
      floorplanImage: overlayDataUrl,
    }
  }, [borderPoints, measurements, isBorderClosed, spacing, unitDisplay, overlayDataUrl])

  useEffect(() => {
    if (!payload) {
      setShowPayload(false)
    }
  }, [payload])

  useEffect(() => {
    setPayloadWarning(null)
  }, [borderPoints, measurements, isBorderClosed])

  useEffect(() => {
    setInitialGraphicsState({ status: 'idle', graphics: [] })
  }, [borderPoints, measurements, spacing, unitDisplay, isBorderClosed, overlayDataUrl])

  const handlePreviewPayload = () => {
    if (!payload) {
      setPayloadWarning('Add at least three border vertices (close the border) and three measurement points before previewing the payload.')
      setShowPayload(false)
      return
    }
    setPayloadWarning(null)
    setShowPayload((previous) => !previous)
  }

  const handleRunInitialGraphics = async () => {
    if (!payload) {
      setPayloadWarning('Provide a closed border and at least three measurement points before running the initial graphics preview.')
      setInitialGraphicsState({ status: 'idle', graphics: [] })
      return
    }

    setPayloadWarning(null)
    setInitialGraphicsState({ status: 'loading', graphics: [] })

  try {
    console.info('[fp1] calling initial graphics endpoint', FAME_API_ENDPOINT, payload)
    const response = await fetch(FAME_API_ENDPOINT, {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error('Service responded with status ' + response.status)
    }

    const responseBody = await response.json()
    console.info('[fp1] initial graphics response body', responseBody)
    const graphics: InitialGraphic[] = []

    if (Array.isArray(responseBody?.graphics)) {
      responseBody.graphics.forEach((item: { name?: string; image?: string }, index: number) => {
        if (!item?.image || typeof item.image !== 'string') {
          return
        }
        const labelKey = item.name ?? 'Layer ' + (index + 1)
        const resolvedName = GRAPHIC_LABELS[labelKey] ?? labelKey
        const imageSrc = item.image.startsWith('data:')
          ? item.image
          : 'data:image/png;base64,' + item.image
        graphics.push({ name: resolvedName, image: imageSrc })
      })
    } else if (responseBody?.images && typeof responseBody.images === 'object') {
      Object.entries(responseBody.images as Record<string, unknown>).forEach(
        ([key, value], index) => {
          if (typeof value !== 'string') {
            return
          }
          const resolvedName = GRAPHIC_LABELS[key] ?? 'Layer ' + (index + 1)
          const imageSrc = value.startsWith('data:')
            ? value
            : 'data:image/png;base64,' + value
          graphics.push({ name: resolvedName, image: imageSrc })
        },
      )
    }

    if (graphics.length === 0) {
      throw new Error('No graphics returned from the analysis service.')
    }

    setInitialGraphicsState({
      status: 'ready',
      graphics,
      message: 'Run is updated',
    })
    } catch (error) {
      setInitialGraphicsState({
        status: 'error',
        graphics: [],
        error: error instanceof Error ? error.message : 'Unknown error',
      })
    }
  }

  const payloadJson = payload ? JSON.stringify(payload, null, 2) : ''

  const borderReady = isBorderClosed && borderPoints.length >= 3
  const measurementReady = measurements.length >= 3
  const hasMeasurements = measurements.length > 0

  let analysisStatus = 'Ready to run analysis'
  if (!borderReady && !hasMeasurements) {
    analysisStatus = 'Add border vertices (close the border) and measurement points'
  } else if (!borderReady) {
    analysisStatus = 'Add at least three border vertices and close the border'
  } else if (!measurementReady) {
    analysisStatus = hasMeasurements
      ? 'Capture at least three measurement points to prepare for analysis'
      : 'Add measurement points'
  }

  const payloadButtonLabel = showPayload ? 'Hide Payload Preview' : 'Preview Payload'
  const modeLabel = mode === 'border' ? 'Border' : 'Measure'
  const canCloseBorder = borderPoints.length >= 3 && !isBorderClosed
  const canUndoBorder = borderPoints.length > 0
  const canRemoveMeasurement = measurements.length > 0

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
        <span className="fp-hint">{overlaySrc ? 'Overlay applied' : 'No overlay'}</span>
        {overlaySrc ? (
          <label className="fp-scale-control">
            <span>Scale {overlayScale.toFixed(2)}x</span>
            <input
              type="range"
              min="0.5"
              max="2.5"
              step="0.05"
              value={overlayScale}
              onChange={(event) => setOverlayScale(Number.parseFloat(event.target.value))}
            />
          </label>
        ) : null}
        <button
          type="button"
          className={mode === 'border' ? 'fp-button is-active' : 'fp-button'}
          onClick={handleToggleMode}
        >
          {mode === 'border' ? 'Border mode on' : 'Border mode off'}
        </button>
        <button
          type="button"
          className="fp-button"
          onClick={handleCloseBorder}
          disabled={!canCloseBorder}
        >
          Close Border
        </button>
        <button
          type="button"
          className="fp-button"
          onClick={handleUndoBorderPoint}
          disabled={!canUndoBorder}
        >
          Undo Border Point
        </button>
        <button
          type="button"
          className="fp-button"
          onClick={handleClearBorder}
          disabled={borderPoints.length === 0}
        >
          Clear Border
        </button>
        <button
          type="button"
          className="fp-button"
          onClick={handleRemoveMeasurement}
          disabled={!canRemoveMeasurement}
        >
          Remove Last Point
        </button>
        <button
          type="button"
          className="fp-button"
          onClick={handleClearMeasurements}
          disabled={measurements.length === 0}
        >
          Clear Points
        </button>
        <button
          type="button"
          className="fp-button"
          onClick={handlePreviewPayload}
        >
          {payloadButtonLabel}
        </button>
      </div>
      <div className="fp-grid">
        <div className="fp-top-labels">
          <div className="fp-corner" />
          {columnTicks.map((tick, index) => (
            <div key={'col-' + index} className="fp-col-label">
              {formatTick(tick)}
            </div>
          ))}
        </div>
        <div className="fp-body">
          <div className="fp-left-labels">
            {rowTicks.map((tick, index) => (
              <div key={'row-' + index} className="fp-row-label">
                {formatTick(tick)}
              </div>
            ))}
          </div>
          <div className="fp-grid-area" onClick={handleCellClick}>
            {overlaySrc ? (
              <img
                src={overlaySrc}
                alt="Floor plan overlay"
                className="fp-overlay"
                style={{ transform: overlayTransform }}
              />
            ) : null}
            <div className="fp-grid-lines" />
            <svg className="fp-border-layer" viewBox="0 0 100 100" preserveAspectRatio="none">
              {borderPoints.length >= 2 ? (
                isBorderClosed && borderPoints.length >= 3 ? (
                  <polygon points={borderSvgPoints} className="fp-border-shape" />
                ) : (
                  <polyline points={borderSvgPoints} className="fp-border-outline" />
                )
              ) : null}
              {borderPoints.map((point, index) => (
                <circle
                  key={'border-' + index}
                  cx={point.nx * 100}
                  cy={point.ny * 100}
                  r="1.2"
                  className="fp-border-node"
                />
              ))}
            </svg>
            {measurements.map((point) => (
              <div
                key={point.id}
                className="fp-point"
                style={{ left: point.nx * 100 + '%', top: point.ny * 100 + '%' }}
                title={point.label + ': ' + point.z.toFixed(2) + ' ' + unitDisplay}
              >
                <span className="fp-point-dot" />
                <span className="fp-point-label">{point.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="fp-analysis-summary">
        <div className="fp-summary-item">
          <strong>Mode:</strong> {modeLabel}
        </div>
        <div className="fp-summary-item">
          <strong>Border vertices:</strong> {borderPoints.length}
        </div>
        <div className="fp-summary-item">
          <strong>Border closed:</strong> {isBorderClosed ? 'Yes' : 'No'}
        </div>
        <div className="fp-summary-item">
          <strong>Measurement points:</strong> {measurements.length}
        </div>
        <div className="fp-summary-item">
          <strong>Spacing:</strong> {formatTick(spacing)}
        </div>
      <div className="fp-summary-item">
        <strong>Analysis status:</strong> {analysisStatus}
      </div>
      <div className="fp-summary-item">
        <strong>API endpoint:</strong> {FAME_API_ENDPOINT_DISPLAY}
      </div>
    </div>
      <div className="fp-run-actions">
        <button
          type="button"
          className="fp-button"
          onClick={handleRunInitialGraphics}
          disabled={!payload || initialGraphicsState.status === 'loading'}
        >
          {initialGraphicsState.status === 'loading' ? 'Generating…' : 'Run Initial Graphics'}
        </button>
        {initialGraphicsState.status === 'loading' ? (
          <span className="fp-hint">Generating preview…</span>
        ) : null}
        {initialGraphicsState.status === 'ready' ? (
          <span className="fp-hint">{initialGraphicsState.message ?? 'Initial graphics ready'}</span>
        ) : null}
        {initialGraphicsState.status === 'error' && initialGraphicsState.error ? (
          <span className="fp-warning">{initialGraphicsState.error}</span>
        ) : null}
      </div>
      {initialGraphicsState.status === 'ready' && initialGraphicsState.graphics.length > 0 ? (
        <div className="fp-graphics-grid">
          {initialGraphicsState.graphics.map((graphic, index) => (
            <figure key={graphic.name + '-' + index} className="fp-graphic-card">
              <img src={graphic.image} alt={graphic.name} />
              <figcaption>{graphic.name}</figcaption>
            </figure>
          ))}
        </div>
      ) : null}
    <div className="fp-data-panels">
      <div className="fp-panel">
        <h4>Measurement Points</h4>
        {measurements.length === 0 ? (
          <p className="fp-hint">Switch to Measure mode and click the grid to add points.</p>
          ) : (
            <ul className="fp-item-list">
              {measurements.map((point) => (
                <li key={point.id}>
                  <strong>{point.label}</strong>
                  <span>X: {point.x.toFixed(2)} {unitDisplay}</span>
                  <span>Y: {point.y.toFixed(2)} {unitDisplay}</span>
                  <span>Z: {point.z.toFixed(2)} {unitDisplay}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="fp-panel">
          <h4>Border Vertices</h4>
          {borderPoints.length === 0 ? (
            <p className="fp-hint">Switch to Border mode to trace the footprint.</p>
          ) : (
            <ul className="fp-item-list">
              {borderPoints.map((point, index) => (
                <li key={point.id}>
                  <strong>Vertex {index + 1}</strong>
                  <span>X: {point.x.toFixed(2)} {unitDisplay}</span>
                  <span>Y: {point.y.toFixed(2)} {unitDisplay}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="fp-payload-actions">
        <button type="button" className="fp-button" onClick={handlePreviewPayload}>
          {payloadButtonLabel}
        </button>
        {payloadWarning ? <span className="fp-warning">{payloadWarning}</span> : null}
      </div>
      {showPayload && payload ? (
        <pre className="fp-payload-preview">{payloadJson}</pre>
      ) : null}
    </div>
  )
}


export default App
