import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Sidebar } from "./components/layout/Sidebar";
import { OptimizationRun } from "./pages/OptimizationRun";
import { RunHistory } from "./pages/RunHistory";
import { useWebSocket } from "./hooks/useWebSocket";

export default function App() {
  useWebSocket();

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-950 text-gray-100">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<OptimizationRun />} />
            <Route path="/history" element={<RunHistory />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
