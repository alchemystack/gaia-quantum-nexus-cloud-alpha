import { useState, useEffect, useRef, useCallback } from 'react';
import type { GenerationRequest, WebSocketMessage, TokenResponse } from '../types/quantum';

interface UseWebSocketProps {
  url: string;
  onToken?: (response: TokenResponse) => void;
  onComplete?: (sessionId: string, totalTokens: number) => void;
  onError?: (error: string) => void;
}

export function useWebSocket({ url, onToken, onComplete, onError }: UseWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 10;

  const connect = useCallback(() => {
    // Clean up existing connection first
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) return;
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (isConnecting) return; // Prevent multiple simultaneous connection attempts
    
    setIsConnecting(true);
    
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${url}`;
    
    console.log('Attempting WebSocket connection to:', wsUrl);
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
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(request));
      setIsGenerating(true);
      return true;
    }
    return false;
  }, []);

  const stopGeneration = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    setIsGenerating(false);
  }, []);

  useEffect(() => {
    connect();
    
    return () => {
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
