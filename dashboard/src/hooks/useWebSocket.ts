import { useEffect } from "react";
import { useRunStore } from "../stores/useRunStore";
import type { WSMessage } from "../types/api";

export function useWebSocket() {
  const handleMessage = useRunStore((s) => s.handleMessage);
  const setStatus = useRunStore((s) => s.setStatus);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | undefined;
    let reconnectDelay = 1000;
    let destroyed = false;

    function connect() {
      if (destroyed) return;

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${window.location.host}/ws/optimization`;

      setStatus("connecting");
      ws = new WebSocket(url);

      ws.onopen = () => {
        reconnectDelay = 1000;
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg: WSMessage = JSON.parse(event.data as string);
          handleMessage(msg);
        } catch (e) {
          console.error("Failed to parse WS message:", e);
        }
      };

      ws.onclose = () => {
        ws = null;
        if (destroyed) return;
        const currentStatus = useRunStore.getState().status;
        // Don't reconnect if run already completed — server has shut down
        if (currentStatus === "complete") return;
        // If we had data but lost connection, mark as complete (server exited)
        if (currentStatus === "running" && useRunStore.getState().report) {
          setStatus("complete");
          return;
        }
        // Auto-reconnect with exponential backoff
        reconnectTimeout = setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 2, 10000);
          connect();
        }, reconnectDelay);
      };

      ws.onerror = () => {
        ws?.close();
      };
    }

    connect();

    return () => {
      destroyed = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, [handleMessage, setStatus]);
}
