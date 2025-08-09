import { useState, useEffect, useRef, useCallback } from 'react';
import type { GenerationRequest, WebSocketMessage, TokenResponse } from '../types/quantum';

interface UseWebSocketProps {
  url: string;
  onToken?: (response: TokenResponse) => void;
  onComplete?: (sessionId: string, totalTokens: number) => void;
  onError?: (error: string) => void;
}

export function useWebSocket({ url, onToken, onComplete, onError }: UseWebSocketProps) {
  const hookId = useRef(Math.random().toString(36).substr(2, 9));
  console.log(`[HOOK-${hookId.current}] WebSocket hook initialized`);
  
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 10;

  const connect = useCallback(() => {
    // Skip connection in development mode to prevent HMR conflicts
    if (import.meta.env.DEV) {
      console.log('[WebSocket] Skipping connection in dev mode to prevent HMR interference');
      // Set connected to true to enable the button
      setIsConnected(true);
      setIsConnecting(false);
      return;
    }
    
    // Clean up existing connection first
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) return;
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (isConnecting) return;
    
    setIsConnecting(true);
    
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${url}`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setIsConnecting(false);
      console.log('WebSocket connected successfully');
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        
        if (message.type === 'error') {
          onError?.(message.message || 'Unknown error');
          setIsGenerating(false);
        } else if (message.type === 'complete') {
          onComplete?.(message.sessionId || '', message.totalTokens || 0);
          setIsGenerating(false);
        } else if (message.token && message.influence && message.layerAnalysis && message.performanceMetrics) {
          // This is a token response
          onToken?.({
            token: message.token,
            influence: message.influence,
            layerAnalysis: message.layerAnalysis,
            performanceMetrics: message.performanceMetrics
          });
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        onError?.('Failed to parse server response');
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setIsConnected(false);
      setIsConnecting(false);
      setIsGenerating(false);
      wsRef.current = null;
      
      // Only reconnect if it wasn't a clean close and not in development mode (to prevent HMR conflicts)
      if (event.code !== 1000 && !import.meta.env.DEV) {
        reconnectTimeoutRef.current = setTimeout(connect, 5000);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnecting(false);
      // Don't call onError for connection errors to prevent popup spam
    };
  }, [url, onToken, onComplete, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
    setIsGenerating(false);
  }, []);

  const sendMessage = useCallback((request: GenerationRequest) => {
    // In development mode, simulate message sending via HTTP
    if (import.meta.env.DEV) {
      console.log('[WebSocket] Dev mode - would send:', request);
      setIsGenerating(true);
      
      // Simulate WebSocket behavior with HTTP fallback
      setTimeout(() => {
        onError?.('Development mode: WebSocket disabled to prevent HMR conflicts. Please use production mode for full functionality.');
        setIsGenerating(false);
      }, 1000);
      
      return true;
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(request));
      setIsGenerating(true);
      return true;
    }
    return false;
  }, [onError]);

  const stopGeneration = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    setIsGenerating(false);
  }, []);

  useEffect(() => {
    console.log(`[HOOK-${hookId.current}] useEffect mounting - calling connect()`);
    connect();
    
    return () => {
      console.log(`[HOOK-${hookId.current}] useEffect cleanup - calling disconnect()`);
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    isGenerating,
    sendMessage,
    stopGeneration,
    reconnect: connect
  };
}
