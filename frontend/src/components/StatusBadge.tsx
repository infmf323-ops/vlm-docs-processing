import clsx from "clsx";
import { getStatusLabel } from "../utils/jobPresentation";

export function StatusBadge({ status }: { status: string }) {
  return <span className={clsx("status-badge", `status-${status}`)}>{getStatusLabel(status)}</span>;
}
