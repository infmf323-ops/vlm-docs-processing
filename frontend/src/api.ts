import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export type LoginPayload = { email: string; password: string };
export type RegisterPayload = { email: string; full_name: string; password: string };

export type DocumentType =
  | "invoice"
  | "passport"
  | "id_card"
  | "driver_license"
  | "financial_statement"
  | "other";

export type ExtractionEngine = "donut" | "paddleocr_vl" | "qwen2_5_vl" | "custom";

export type JobListItem = {
  id: number;
  status: "queued" | "processing" | "done" | "failed";
  original_filename: string;
  created_at: string;
  updated_at?: string | null;
  finished_at?: string | null;
  error_message?: string | null;
};

export type JobDetail = JobListItem & {
  content_type: string;
  page_count: number;
  requested_document_type?: DocumentType | null;
  requested_engine?: ExtractionEngine | null;
  preview_url?: string | null;
  normalized_result?: Record<string, unknown> | null;
  raw_result?: Record<string, unknown> | null;
  download_url?: string | null;
};

export type JobStatus = JobListItem["status"];

export type JobStats = {
  total: number;
  by_status: Record<string, number>;
  by_document_type: Record<string, number>;
  mrz_total: number;
  mrz_valid: number;
  mrz_valid_rate?: number | null;
  success_rate?: number | null;
};

export type JobFilters = {
  status?: JobStatus;
  documentType?: DocumentType;
};

export const authApi = {
  login: async (payload: LoginPayload) => (await api.post("/auth/login", payload)).data,
  register: async (payload: RegisterPayload) =>
    (await api.post("/auth/register", payload)).data,
};

function filterParams(filters?: JobFilters) {
  const params: Record<string, string> = {};
  if (filters?.status) params.status = filters.status;
  if (filters?.documentType) params.document_type = filters.documentType;
  return params;
}

export const jobsApi = {
  list: async (filters?: JobFilters): Promise<JobListItem[]> =>
    (await api.get("/jobs", { params: filterParams(filters) })).data,
  detail: async (id: string): Promise<JobDetail> => (await api.get(`/jobs/${id}`)).data,
  stats: async (): Promise<JobStats> => (await api.get("/jobs/stats")).data,
  upload: async (
    file: File,
    options?: {
      documentType?: DocumentType;
      extractionEngine?: ExtractionEngine;
    }
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    if (options?.documentType) {
      formData.append("document_type", options.documentType);
    }
    if (options?.extractionEngine) {
      formData.append("extraction_engine", options.extractionEngine);
    }
    return (await api.post("/jobs", formData)).data;
  },
  retry: async (id: number) => (await api.post(`/jobs/${id}/retry`)).data,
  remove: async (id: number) => (await api.delete(`/jobs/${id}`)).data,
  exportCsv: async (filters?: JobFilters): Promise<Blob> =>
    (await api.get("/jobs/export.csv", { params: filterParams(filters), responseType: "blob" }))
      .data,
};
