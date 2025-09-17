import { useState, useRef, ChangeEvent } from "react"
import "./FloorplanSheet.css"

interface FloorplanSheetProps {
  getValue: (cell: string) => string
  onChange: (cell: string, value: string) => void
  onSelect: (cell: string) => void
}

export function FloorplanSheet({ getValue, onChange, onSelect }: FloorplanSheetProps) {
  const [floorplanImage, setFloorplanImage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setFloorplanImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  // Generate column labels: A-Z, AA-AZ, BA-BZ, CA-CY (total 103 columns)
  const columns = []
  // Single letters A-Z (26 columns)
  for (let i = 0; i < 26; i++) {
    columns.push(String.fromCharCode(65 + i))
  }
  // AA-AZ (26 columns)
  for (let i = 0; i < 26; i++) {
    columns.push('A' + String.fromCharCode(65 + i))
  }
  // BA-BZ (26 columns)
  for (let i = 0; i < 26; i++) {
    columns.push('B' + String.fromCharCode(65 + i))
  }
  // CA-CY (25 columns to match the screenshot)
  for (let i = 0; i < 25; i++) {
    columns.push('C' + String.fromCharCode(65 + i))
  }

  const rows = Array.from({ length: 100 }, (_, i) => i + 1)

  return (
    <div className="floorplan-container">
      <div className="floorplan-toolbar">
        <button onClick={handleUploadClick} className="upload-button">
          Upload Floorplan
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: "none" }}
        />
        {floorplanImage && (
          <button onClick={() => setFloorplanImage(null)} className="clear-button">
            Clear Image
          </button>
        )}
      </div>
      <div className="floorplan-grid-wrapper">
        {floorplanImage && (
          <img
            src={floorplanImage}
            alt="Floorplan overlay"
            className="floorplan-overlay"
          />
        )}
        <table className="floorplan-grid">
          <thead>
            <tr>
              <th className="floorplan-corner"></th>
              {columns.map((col) => (
                <th key={col} className="floorplan-column-header">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row}>
                <th className="floorplan-row-header">{row}</th>
                {columns.map((col) => {
                  const cellRef = `${col}${row}`
                  const value = getValue(cellRef)

                  return (
                    <td
                      key={col}
                      className="floorplan-cell"
                      onClick={() => onSelect(cellRef)}
                    >
                      {value}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}