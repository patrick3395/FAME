interface SetupSheetProps {
  getValue: (cell: string) => string
  onChange: (cell: string, value: string) => void
  onSelect: (cell: string) => void
  getValidation: (cell: string) => { options: string[]; allowBlank: boolean }
}

export function SetupSheet({ getValue, onChange, onSelect, getValidation }: SetupSheetProps) {
  const type = getValue("C2")
  const version = getValue("F3")
  const engCode = getValue("G2")
  const stage = getValue("I3")
  const address = getValue("E6")
  const badgeLabel = getValue("E8")
  const badgeDate = getValue("G8")
  const badgeTitle = getValue("E9")
  const note = getValue("B14")

  const typeValidation = getValidation("C2")
  const versionValidation = getValidation("F3")
  const addressValidation = getValidation("E6")

  const renderSelect = (
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

  return (
    <div className="setup-sheet">
      <div className="setup-grid">
        <div className="setup-field">
          <label>Type (C2)</label>
          {renderSelect(type, "C2", typeValidation)}
        </div>
        <div className="setup-field">
          <label>Version (F3)</label>
          {renderSelect(version, "F3", versionValidation)}
        </div>
        <div className="setup-field">
          <label>Eng. Code (G2)</label>
          <input
            value={engCode}
            onFocus={() => onSelect("G2")}
            onChange={(event) => onChange("G2", event.target.value)}
          />
        </div>
        <div className="setup-field">
          <label>Stage (I3)</label>
          <input
            value={stage}
            onFocus={() => onSelect("I3")}
            onChange={(event) => onChange("I3", event.target.value)}
          />
        </div>
        <div className="setup-field">
          <label>Address (E6)</label>
          {renderSelect(address, "E6", addressValidation)}
        </div>
        <div className="setup-field">
          <label>Badge Label (E8)</label>
          <input
            value={badgeLabel}
            onFocus={() => onSelect("E8")}
            onChange={(event) => onChange("E8", event.target.value)}
          />
        </div>
        <div className="setup-field">
          <label>Badge Date (G8)</label>
          <input
            value={badgeDate}
            onFocus={() => onSelect("G8")}
            onChange={(event) => onChange("G8", event.target.value)}
          />
        </div>
        <button type="button" className="setup-badge" onClick={() => onSelect("E9")}>
          {badgeTitle}
        </button>
        <button type="button" className="setup-note" onClick={() => onSelect("B14")}>
          {note}
        </button>
      </div>
    </div>
  )
}