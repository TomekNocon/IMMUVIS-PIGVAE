import { useState, type DragEvent } from 'react'
import { MAX_UPLOAD_BYTES, USING_MOCK, type InfoResponse } from '@/api'

export interface UploadError {
  title: string
  detail: string
  retry?: boolean
}

interface Props {
  info: InfoResponse | undefined
  serverReady: boolean
  file: File | null
  error: UploadError | null
  onFile: (file: File | null, error: UploadError | null) => void
  onAnalyse: (file: File) => void
}

const mb = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MB`

function check(f: File): UploadError | null {
  if (!/\.(tiff?|npy)$/i.test(f.name)) {
    return { title: "This file type isn't supported", detail: 'Use a .tif, .tiff or .npy file.' }
  }
  if (f.size > MAX_UPLOAD_BYTES) {
    return { title: 'File is larger than 300 MB', detail: `${f.name} · ${(f.size / 1048576).toFixed(0)} MB` }
  }
  return null
}

// Mock mode only: the mock picks its result from the file name, so an empty named file is enough.
const exampleFile = (name: string) => new File([new Uint8Array(1)], name, { type: 'image/tiff' })

export function Upload({ info, serverReady, file, error, onFile, onAnalyse }: Props) {
  const [dragging, setDragging] = useState(false)
  const [markersOpen, setMarkersOpen] = useState(false)

  const pick = (f: File | undefined) => {
    if (!f) return
    const err = check(f)
    onFile(err ? null : f, err)
  }
  const drop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    pick(e.dataTransfer.files[0])
  }
  const [classA, classS] = info?.classes ?? ['Adenocarcinoma', 'Squamous cell carcinoma']
  const nMarkers = info?.n_markers ?? 43

  return (
    <main
      className="rise"
      style={{
        flex: 1, width: '100%', maxWidth: 1120, margin: '0 auto', padding: '48px 28px', display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))', gap: 48, alignItems: 'start',
      }}
    >
      <section style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
        <h1 className="serif" style={{ fontWeight: 400, fontSize: 44, lineHeight: 1.08, margin: 0, textWrap: 'pretty' }}>
          Analyse an IMC image
        </h1>
        <p className="mut" style={{ margin: 0, maxWidth: '52ch' }}>
          Upload one raw imaging mass cytometry image of lung tumour tissue. The model predicts the subtype and
          shows which {info?.tile_size ?? 128} × {info?.tile_size ?? 128} px crops drove the answer.
        </p>
        <label
          className={`dropzone${dragging ? ' dragging' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
        >
          <input
            type="file"
            accept=".tif,.tiff,.npy"
            style={{ display: 'none' }}
            onChange={(e) => {
              pick(e.target.files?.[0])
              e.target.value = ''
            }}
          />
          <span style={{ fontSize: 17, fontWeight: 500 }}>Drop an image here, or choose a file</span>
          <span className="mono mut" style={{ fontSize: 12 }}>
            .tif · .tiff · .npy · up to 300 MB
          </span>
        </label>

        {file && (
          <div
            className="card"
            style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', padding: '12px 14px', animation: 'rise .35s both' }}
          >
            <div style={{ flex: 1, minWidth: 140 }}>
              <div style={{ fontWeight: 500, overflowWrap: 'anywhere' }}>{file.name}</div>
              <div className="mono mut" style={{ fontSize: 12 }}>
                {mb(file.size)}
              </div>
            </div>
            <button className="primary" style={{ height: 44, padding: '0 22px' }} onClick={() => onAnalyse(file)}>
              Analyse
            </button>
          </div>
        )}

        {error && (
          <div
            role="alert"
            style={{
              padding: '14px 16px', border: '1px solid var(--err)', borderRadius: 8, display: 'flex',
              flexDirection: 'column', gap: 6, animation: 'rise .35s both',
            }}
          >
            <div style={{ fontWeight: 600 }}>{error.title}</div>
            <div className="mono mut" style={{ fontSize: 13, overflowWrap: 'anywhere' }}>
              {error.detail}
            </div>
            {error.retry && file && (
              <button className="ghost" style={{ alignSelf: 'flex-start', marginTop: 4 }} onClick={() => onAnalyse(file)}>
                Try again
              </button>
            )}
          </div>
        )}

        {USING_MOCK && (
          <div className="mut" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', fontSize: 13 }}>
            <span>No file to hand?</span>
            <button className="textbtn" style={{ borderBottomColor: 'var(--a)' }} onClick={() => onAnalyse(exampleFile('8080_adeno.tiff'))}>
              {classA.split(' ')[0]} example
            </button>
            <button className="textbtn" style={{ borderBottomColor: 'var(--s)' }} onClick={() => onAnalyse(exampleFile('7810_squam.tiff'))}>
              {classS.split(' ')[0]} example
            </button>
          </div>
        )}
      </section>

      <aside style={{ display: 'flex', flexDirection: 'column', gap: 28, minWidth: 0 }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 10 }}>
            <span className="caps">Model</span>
            <span className="mut" style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 13 }}>
              <span
                style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: serverReady ? 'oklch(0.75 0.14 150)' : 'var(--mut)',
                  animation: serverReady ? 'none' : 'pulse 1.2s infinite',
                }}
              />
              {serverReady ? 'Server ready' : 'Waking server'}
            </span>
          </div>
          <dl
            className="card"
            style={{ margin: 0, display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '9px 20px', padding: 16, fontSize: 14 }}
          >
            <dt className="mut">Task</dt>
            <dd className="mono" style={{ margin: 0 }}>{info?.label ?? '…'}</dd>
            <dt className="mut">Classes</dt>
            <dd style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[classA, classS].map((c, i) => (
                <span key={c} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 9, height: 9, background: i === 0 ? 'var(--a)' : 'var(--s)' }} />
                  {c}
                </span>
              ))}
            </dd>
            <dt className="mut">Fold</dt>
            <dd className="mono" style={{ margin: 0 }}>{info?.fold ?? '…'}</dd>
            <dt className="mut">Validation AUROC</dt>
            <dd className="mono" style={{ margin: 0 }}>{info ? info.fold_val_auroc.toFixed(3) : '…'}</dd>
            <dt className="mut">Bundle</dt>
            <dd className="mono" style={{ margin: 0 }}>{info?.bundle ?? '…'}</dd>
          </dl>
        </div>

        <div>
          <div className="caps" style={{ marginBottom: 10 }}>Expected input</div>
          <p style={{ margin: '0 0 10px', fontSize: 14 }}>
            A raw IMC image with {nMarkers} channels in the cords panel order.
          </p>
          <button className="textbtn" onClick={() => setMarkersOpen((o) => !o)} disabled={!info}>
            {markersOpen ? `Hide the ${nMarkers} markers` : `Show the ${nMarkers} markers`}
          </button>
          <div className={`expand${markersOpen ? ' open' : ''}`}>
            <ol style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {(info?.markers ?? []).map((m, i) => (
                <li
                  key={m}
                  className="mono"
                  style={{ fontSize: 12, padding: '3px 8px', border: '1px solid var(--line)', borderRadius: 4, marginTop: 6 }}
                >
                  <span className="mut">{String(i + 1).padStart(2, '0')}</span> {m}
                </li>
              ))}
            </ol>
          </div>
        </div>
      </aside>
    </main>
  )
}
