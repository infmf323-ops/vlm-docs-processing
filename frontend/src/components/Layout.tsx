import { NavLink, Outlet, useNavigate } from "react-router-dom";

export function Layout() {
  const navigate = useNavigate();

  return (
    <div className="shell">
      <aside className="sidebar">
        <div>
          <div className="brand">Invoice Vision</div>
          <div className="muted">Локальный inference на RTX 3080</div>
        </div>

        <nav className="nav">
          <NavLink to="/dashboard" className="nav-link">
            Дашборд
          </NavLink>
          <NavLink to="/upload" className="nav-link">
            Загрузка
          </NavLink>
          <NavLink to="/jobs" className="nav-link">
            Задания
          </NavLink>
        </nav>

        <button
          className="ghost-button"
          onClick={() => {
            localStorage.removeItem("token");
            navigate("/login");
          }}
        >
          Выйти
        </button>
      </aside>

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
