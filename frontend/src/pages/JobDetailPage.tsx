import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { jobsApi } from "../api";
import { StatusBadge } from "../components/StatusBadge";
import {
  getDocumentTypeLabel,
  getEngineLabel,
  getFieldLabel,
  getFriendlyJobError,
} from "../utils/jobPresentation";

type FieldEntry = {
  key: string;
  label: string;
  value: string;
};

function flattenFields(value: unknown, prefix = ""): FieldEntry[] {
  if (value === null || value === undefined) {
    return [];
  }

  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    if (value.every((item) => typeof item !== "object" || item === null)) {
      return [
        {
          key: prefix,
          label: getFieldLabel(prefix),
          value: value.map((item) => String(item)).join(", "),
        },
      ];
    }

    return value.flatMap((item, index) =>
      flattenFields(item, prefix ? `${prefix}[${index}]` : `[${index}]`)
    );
  }

  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, nested]) =>
      flattenFields(nested, prefix ? `${prefix}.${key}` : key)
    );
  }

  const parts = prefix.split(".");
  const terminalKey = prefix.includes(".") ? parts[parts.length - 1] ?? prefix : prefix;
  return [
    {
      key: prefix,
      label: getFieldLabel(terminalKey),
      value: String(value),
    },
  ];
}

export function JobDetailPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => jobsApi.detail(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: 3000,
  });

  const retryMutation = useMutation({
    mutationFn: () => jobsApi.retry(Number(jobId)),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["job", jobId] });
      await queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => jobsApi.remove(Number(jobId)),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["jobs"] });
      navigate("/jobs");
    },
  });

  const handleDelete = () => {
    if (window.confirm("Удалить задание безвозвратно?")) {
      deleteMutation.mutate();
    }
  };

  const fieldEntries = useMemo(() => {
    const fields = (data?.normalized_result?.fields as Record<string, unknown> | undefined) ?? {};
    return flattenFields(fields).filter((entry) => entry.value.trim().length > 0);
  }, [data]);

  const mrzInfo = useMemo(() => {
    const normalized: any = data?.normalized_result;
    const raw: any = data?.raw_result;
    if (!normalized || raw?.mode !== "passport_mrz") return null;
    return {
      valid: Boolean(normalized.validation?.is_valid),
      corrected: Boolean(raw?.corrected),
      nCorrections: Number(raw?.n_corrections ?? 0),
    };
  }, [data]);

  if (isLoading) return <section className="panel">Загружаем детали job...</section>;
  if (error || !data) return <section className="panel error-box">Не удалось загрузить детали.</section>;

  const friendlyError = getFriendlyJobError(data.error_message);

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h1>Job #{data.id}</h1>
          <p className="muted">{data.original_filename}</p>
        </div>
        <div className="job-actions">
          <StatusBadge status={data.status} />
          <button className="ghost-button" onClick={() => retryMutation.mutate()}>
            Перезапустить
          </button>
          <button
            className="ghost-button danger"
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
          >
            Удалить
          </button>
          {data.download_url && (
            <a className="primary-button" href={data.download_url} target="_blank" rel="noreferrer">
              Скачать JSON
            </a>
          )}
        </div>
      </div>

      {(data.status === "queued" || data.status === "processing") && (
        <div className="processing-bar" aria-label="Документ обрабатывается" />
      )}

      {friendlyError && <div className="error-box">{friendlyError}</div>}

      <div className="detail-grid">
        <div className="preview-card">
          <div className="meta-grid">
            <div className="field-pair">
              <span className="field-key">Запрошенный тип документа</span>
              <span className="field-value">{getDocumentTypeLabel(data.requested_document_type)}</span>
            </div>
            {data.page_count > 1 && (
              <div className="field-pair">
                <span className="field-key">Страниц в документе</span>
                <span className="field-value">{data.page_count}</span>
              </div>
            )}
            <div className="field-pair">
              <span className="field-key">Запрошенный движок</span>
              <span className="field-value">{getEngineLabel(data.requested_engine)}</span>
            </div>
            <div className="field-pair">
              <span className="field-key">Распознанный тип</span>
              <span className="field-value">
                {getDocumentTypeLabel((data.normalized_result?.document_type as string | undefined) ?? null)}
              </span>
            </div>
          </div>

          {data.preview_url ? (
            <img src={data.preview_url} alt="Превью документа" className="preview-image" />
          ) : (
            <div className="muted">Превью пока недоступно</div>
          )}
        </div>

        <div className="fields-card">
          {mrzInfo ? (
            <div className={mrzInfo.valid ? "mrz-badge mrz-ok" : "mrz-badge mrz-warn"}>
              {mrzInfo.valid
                ? "MRZ корректна — контрольные цифры пройдены"
                : "MRZ не прошла проверку контрольными цифрами"}
              {mrzInfo.corrected
                ? " · исправлено ошибок распознавания: " + mrzInfo.nCorrections
                : ""}
            </div>
          ) : null}
          <div className="field-section">
            <h3>Извлеченные поля</h3>
            <div className="field-grid">
              {fieldEntries.length > 0 ? (
                fieldEntries.map((entry) => (
                  <div key={entry.key} className="field-pair">
                    <span className="field-key">{entry.label}</span>
                    <span className="field-value">{entry.value}</span>
                  </div>
                ))
              ) : (
                <div className="muted">Пока нет заполненных полей.</div>
              )}
            </div>
          </div>

          <details className="json-block">
            <summary>Раскрыть JSON</summary>
            <pre>{JSON.stringify(data.normalized_result, null, 2)}</pre>
          </details>

          <details className="json-block">
            <summary>Показать raw output модели</summary>
            <pre>{JSON.stringify(data.raw_result, null, 2)}</pre>
          </details>

          {data.error_message && data.error_message !== friendlyError && (
            <details className="json-block">
              <summary>Показать техническую ошибку</summary>
              <pre>{data.error_message}</pre>
            </details>
          )}
        </div>
      </div>
    </section>
  );
}
  