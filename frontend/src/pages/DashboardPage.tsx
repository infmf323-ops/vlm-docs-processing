import { useQuery } from "@tanstack/react-query";
import { jobsApi } from "../api";
import { getStatusLabel, getDocumentTypeLabel } from "../utils/jobPresentation";

function percent(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

const STATUS_ORDER = ["queued", "processing", "done", "failed"];
const STATUS_COLOR: Record<string, string> = {
  queued: "#f0cc7c",
  processing: "#89b8ff",
  done: "#5ef2c0",
  failed: "#ff9b9b",
};

type BarRow = { label: string; value: number; color?: string };

function BarChart({ rows }: { rows: BarRow[] }) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  return (
    <div className="bar-chart">
      {rows.map((row) => (
        <div key={row.label} className="bar-row">
          <span className="bar-label">{row.label}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: `${(row.value / max) * 100}%`,
                background: row.color ?? "#67a0ff",
              }}
            />
          </div>
          <span className="bar-value">{row.value}</span>
        </div>
      ))}
    </div>
  );
}

export function DashboardPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs", "stats"],
    queryFn: jobsApi.stats,
    refetchInterval: 5000,
  });

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h1>Дашборд</h1>
          <p className="muted">Сводка по заданиям обновляется автоматически.</p>
        </div>
      </div>

      {isLoading && <div className="muted">Загружаем статистику...</div>}
      {error && <div className="error-box">Не удалось загрузить статистику.</div>}

      {data && (
        <>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{data.total}</div>
              <div className="stat-label">Всего заданий</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{percent(data.success_rate)}</div>
              <div className="stat-label">Доля успешных</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{data.mrz_total}</div>
              <div className="stat-label">Паспортов по MRZ</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{percent(data.mrz_valid_rate)}</div>
              <div className="stat-label">MRZ прошли контроль</div>
            </div>
          </div>

          <div className="detail-grid">
            <div className="fields-card">
              <h3>По статусам</h3>
              {Object.keys(data.by_status).length === 0 ? (
                <div className="muted">Пока нет заданий.</div>
              ) : (
                <BarChart
                  rows={STATUS_ORDER.filter((s) => data.by_status[s]).map((s) => ({
                    label: getStatusLabel(s),
                    value: data.by_status[s],
                    color: STATUS_COLOR[s],
                  }))}
                />
              )}
            </div>

            <div className="fields-card">
              <h3>По типам документов</h3>
              {Object.keys(data.by_document_type).length === 0 ? (
                <div className="muted">Нет обработанных документов.</div>
              ) : (
                <BarChart
                  rows={Object.entries(data.by_document_type).map(([type, count]) => ({
                    label: getDocumentTypeLabel(type),
                    value: count,
                  }))}
                />
              )}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
