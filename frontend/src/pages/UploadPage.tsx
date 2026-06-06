import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { jobsApi, type DocumentType, type ExtractionEngine } from "../api";

const DOCUMENT_TYPES: { value: DocumentType; label: string }[] = [
  { value: "invoice", label: "Счет / Invoice" },
  { value: "passport", label: "Паспорт" },
  { value: "id_card", label: "ID-карта" },
  { value: "driver_license", label: "Водительское удостоверение" },
  { value: "financial_statement", label: "Финансовая выписка" },
  { value: "other", label: "Другой документ" },
];

const EXTRACTION_ENGINES: { value: ExtractionEngine; label: string }[] = [
  { value: "donut", label: "Donut" },
  { value: "paddleocr_vl", label: "PaddleOCR-VL" },
  { value: "qwen2_5_vl", label: "Qwen2.5-VL" },
];

const DEFAULT_ENGINE_BY_DOCUMENT_TYPE: Record<DocumentType, ExtractionEngine> = {
  invoice: "donut",
  passport: "paddleocr_vl",
  id_card: "paddleocr_vl",
  driver_license: "paddleocr_vl",
  financial_statement: "paddleocr_vl",
  other: "paddleocr_vl",
};

export function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<DocumentType>("invoice");
  const [extractionEngine, setExtractionEngine] = useState<ExtractionEngine>(
    DEFAULT_ENGINE_BY_DOCUMENT_TYPE.invoice,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await jobsApi.upload(file, { documentType, extractionEngine });
      navigate(`/jobs/${result.id}`);
    } catch {
      setError("Не удалось создать job. Проверьте файл, тип документа и выбранный движок.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h1>Загрузка документа</h1>
          <p className="muted">
            Поддерживаются изображения и многостраничные PDF — обрабатываются все страницы.
          </p>
        </div>
      </div>

      <div className="upload-box">
        <div className="field">
          <label htmlFor="document-type">Тип документа</label>
          <select
            id="document-type"
            value={documentType}
            onChange={(e) => {
              const nextDocumentType = e.target.value as DocumentType;
              setDocumentType(nextDocumentType);
              setExtractionEngine(DEFAULT_ENGINE_BY_DOCUMENT_TYPE[nextDocumentType]);
            }}
          >
            {DOCUMENT_TYPES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="extraction-engine">Движок извлечения</label>
          <select
            id="extraction-engine"
            value={extractionEngine}
            onChange={(e) => setExtractionEngine(e.target.value as ExtractionEngine)}
          >
            {EXTRACTION_ENGINES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <input
          type="file"
          accept=".png,.jpg,.jpeg,.pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file && <div className="muted">Выбран файл: {file.name}</div>}
        {error && <div className="error-box">{error}</div>}
        <button className="primary-button" disabled={!file || loading} onClick={onUpload}>
          {loading ? "Создаем job..." : "Запустить обработку"}
        </button>
      </div>
    </section>
  );
}
