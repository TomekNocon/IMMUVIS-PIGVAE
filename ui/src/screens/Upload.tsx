import { useState, type DragEvent } from 'react'
import { MAX_UPLOAD_BYTES, USING_MOCK, type InfoResponse } from '@/api'
import { cx } from '@/components/cx'
import './Upload.css'

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
  const pick = (f: File | undefined) => {
    if (!f) return
    const err = check(f)
    onFile(err ? null : f, err)
  }
  const classes = info?.classes ?? ['Adenocarcinoma', 'Squamous cell carcinoma']
  const tile = info?.tile_size ?? 128

  return (
    <main className="rise upload">
      <section className="uploadMain">
        <h1 className="title">Analyse an IMC image</h1>
        <p className="mut uploadLead">
          Upload one raw imaging mass cytometry image of lung tumour tissue. The model predicts the subtype and
          shows which {tile} × {tile} px crops drove the answer.
        </p>
        <Dropzone onPick={pick} />
        {file && <SelectedFile file={file} onAnalyse={() => onAnalyse(file)} />}
        {error && <ErrorBox error={error} onRetry={file ? () => onAnalyse(file) : undefined} />}
        {USING_MOCK && (
          <div className="mut examples">
            <span>No file to hand?</span>
            <button className="textbtn toA" onClick={() => onAnalyse(exampleFile('8080_adeno.tiff'))}>
              {classes[0].split(' ')[0]} example
            </button>
            <button className="textbtn toS" onClick={() => onAnalyse(exampleFile('7810_squam.tiff'))}>
              {classes[1].split(' ')[0]} example
            </button>
          </div>
        )}
      </section>

      <aside className="uploadSide">
        <ModelCard info={info} classes={classes} serverReady={serverReady} />
        <ExpectedInput info={info} />
      </aside>
    </main>
  )
}

function Dropzone({ onPick }: { onPick: (f: File | undefined) => void }) {
  const [dragging, setDragging] = useState(false)
  const drop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    onPick(e.dataTransfer.files[0])
  }
  return (
    <label
      className={cx('dropzone', dragging && 'dragging')}
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
        hidden
        onChange={(e) => {
          onPick(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      <span className="dropTitle">Drop an image here, or choose a file</span>
      <span className="hint">.tif · .tiff · .npy · up to 300 MB</span>
    </label>
  )
}

function SelectedFile({ file, onAnalyse }: { file: File; onAnalyse: () => void }) {
  return (
    <div className="card fileCard">
      <div className="fileInfo">
        <div className="fileName">{file.name}</div>
        <div className="hint">{mb(file.size)}</div>
      </div>
      <button className="primary compact" onClick={onAnalyse}>
        Analyse
      </button>
    </div>
  )
}

function ErrorBox({ error, onRetry }: { error: UploadError; onRetry?: () => void }) {
  return (
    <div role="alert" className="errorBox">
      <div className="errorTitle">{error.title}</div>
      <div className="mono mut errorDetail">{error.detail}</div>
      {error.retry && onRetry && (
        <button className="ghost" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

function ModelCard({ info, classes, serverReady }: { info?: InfoResponse; classes: string[]; serverReady: boolean }) {
  return (
    <div>
      <div className="modelHead">
        <span className="caps">Model</span>
        <span className="mut serverStatus">
          <span className={cx('statusDot', serverReady && 'ready')} />
          {serverReady ? 'Server ready' : 'Waking server'}
        </span>
      </div>
      <dl className="card modelCard">
        <dt className="mut">Task</dt>
        <dd className="mono">{info?.label ?? '…'}</dd>
        <dt className="mut">Classes</dt>
        <dd className="classList">
          {classes.slice(0, 2).map((c, i) => (
            <span key={c}>
              <span className={cx('swatch', i === 0 ? 'toA' : 'toS')} />
              {c}
            </span>
          ))}
        </dd>
        <dt className="mut">Fold</dt>
        <dd className="mono">{info?.fold ?? '…'}</dd>
        <dt className="mut">Validation AUROC</dt>
        <dd className="mono">{info ? info.fold_val_auroc.toFixed(3) : '…'}</dd>
        <dt className="mut">Bundle</dt>
        <dd className="mono">{info?.bundle ?? '…'}</dd>
      </dl>
    </div>
  )
}

function ExpectedInput({ info }: { info?: InfoResponse }) {
  const [open, setOpen] = useState(false)
  const n = info?.n_markers ?? 43
  return (
    <div>
      <div className="caps uploadHead">Expected input</div>
      <p className="uploadText">A raw IMC image with {n} channels in the cords panel order.</p>
      <button className="textbtn" onClick={() => setOpen((o) => !o)} disabled={!info}>
        {open ? `Hide the ${n} markers` : `Show the ${n} markers`}
      </button>
      <div className={cx('expand', open && 'open')}>
        <ol className="markerList">
          {(info?.markers ?? []).map((m, i) => (
            <li key={m} className="mono">
              <span className="mut">{String(i + 1).padStart(2, '0')}</span> {m}
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}
