import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type MouseEvent } from "react";
import { Link } from "react-router-dom";
import { jobsApi, type DocumentType, type JobStatus } from "../api";
import { StatusBadge } from "../components/StatusBadge";
import {
  getFriendlyJobError,
  getStatusLabel,
  getDocumentTypeLabel,
} from "../utils/jobPresentation";

const STATUS_OPTIONS: JobStatus[] = ["queued", "processing", "done", "failed"];
const DOCUMENT_TYPES: DocumentType[] = [
  "invoice",
  "passport",
  "id_card",
  "driver_license",
  "financial_statement",
  "other",
];

export function JobsListPage() {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<JobStatus | "">("");
  const [documentType, setDocumentType] = useState<DocumentType | "">("");

  const filters = {
    status: status || undefined,
    documentType: documentType || undefined,
  };

  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs", status, documentType],
    queryFn: () => jobsApi.list(filters),
    refetchInterval: 3000,
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => jobsApi.remove(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const handleExport = async () => {
    const blob = await jobsApi.exportCsv(filters);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "jobs.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDelete = (event: MouseEvent, id: number) => {
    event.preventDefault();
    event.stopPropagation();
    if (window.confirm("Удалить задание безвозвратно?")) {
      deleteMutation.mutate(id);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h1>Список заданий</h1>
          <p className="muted">Статусы обновляются автоматически.</p>
        </div>
        <button className="primary-button" onClick={handleExport}>
          Экспорт в CSV
        </button>
      </div>

      <div className="filters-row">
        <label className="filter">
          <span className="muted">Статус</span>
          <select value={status} onChange={(e) => setStatus(e.target.value as JobStatus | "")}>
            <option value="">Все</option>
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {getStatusLabel(s)}
              </option>
            ))}
          </select>
        </label>
        <label className="filter">
          <span className="muted">Тип документа</span>
          <select
            value={documentType}
            onChange={(e) => setDocumentType(e.target.value as DocumentType | "")}
          >
            <option value="">Все</option>
            {DOCUMENT_TYPES.map((t) => (
              <option key={t} value={t}>
                {getDocumentTypeLabel(t)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {isLoading && <div className="muted">Загружаем список...</div>}
      {error && <div className="error-box">Не удалось загрузить задания.</div>}

      <div className="jobs-list">
        {data?.length === 0 && !isLoading && (
          <div className="muted">Заданий по выбранным фильтрам нет.</div>
        )}
        {data?.map((job) => (
          <Link key={job.id} to={`/jobs/${job.id}`} className="job-card">
            <div className="job-card-header">
              <strong>{job.original_filename}</strong>
              <StatusBadge status={job.status} />
            </div>
            <div className="muted">Задание #{job.id}</div>
            <div className="muted">
              Создано: {new Date(job.created_at).toLocaleString("ru-RU")}
            </div>
            {job.error_message && (
              <div className="error-inline">{getFriendlyJobError(job.error_message)}</div>
            )}
            <button
              className="ghost-button danger"
              onClick={(e) => handleDelete(e, job.id)}
              disabled={deleteMutation.isPending}
            >
              Удалить
            </button>
          </Link>
        ))}
      </div>
    </section>
  );
}
