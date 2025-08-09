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
    // Skip connection in development mode to prevent HMR conflicts
    if (import.meta.env.DEV) {
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
    // In development mode, use HTTP with streaming
    if (import.meta.env.DEV) {
      setIsGenerating(true);
      
      // Use fetch with streaming response
      console.log('[HTTP] Sending generation request:', request);
      
      fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
      }).then(async response => {
        console.log('[HTTP] Response received:', response.status, response.statusText);
        
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        
        if (!response.body) {
          throw new Error('No response body received');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let tokenCount = 0;

        console.log('[HTTP] Starting to read stream...');

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.log('[HTTP] Stream complete');
              break;
            }

            // Decode the chunk and add to buffer
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete lines
            const lines = buffer.split('\n');
            // Keep the last incomplete line in the buffer
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.trim() === '') continue;
              
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  
                  if (data.type === 'token') {
                    tokenCount++;
                    // Remove the type field before passing to onToken
                    const { type, ...tokenData } = data;
                    console.log(`[HTTP] Token ${tokenCount} received:`, tokenData.token);
                    onToken?.(tokenData as TokenResponse);
                  } else if (data.type === 'complete') {
                    console.log('[HTTP] Generation complete:', data);
                    onComplete?.(data.sessionId, data.totalTokens);
                    setIsGenerating(false);
                  } else if (data.type === 'error') {
                    console.error('[HTTP] Generation error:', data.error);
                    onError?.(data.error);
                    setIsGenerating(false);
                  }
                } catch (e) {
                  console.error('[HTTP] Failed to parse SSE data:', e, line);
                }
              }
            }
          }
          
          // Process any remaining buffer
          if (buffer.trim() && buffer.startsWith('data: ')) {
            try {
              const data = JSON.parse(buffer.slice(6));
              if (data.type === 'complete') {
                onComplete?.(data.sessionId, data.totalTokens);
              }
            } catch (e) {
              console.error('Failed to parse final SSE data:', e);
            }
          }
        } catch (error) {
          console.error('Stream reading error:', error);
          onError?.(`Stream reading failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        } finally {
          setIsGenerating(false);
        }
      }).catch(error => {
        console.error('HTTP generation error:', error);
        onError?.(`Generation failed: ${error instanceof Error ? error.message : 'Connection error'}`);
        setIsGenerating(false);
      });
      
      return true;
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(request));
      setIsGenerating(true);
      return true;
    }
    return false;
  }, [onToken, onComplete, onError]);

  const stopGeneration = useCallback(() => {
    console.log('[HTTP] Stopping generation...');
    
    // In development mode, we can't actually stop the server stream
    // but we can stop processing on client side
    if (import.meta.env.DEV) {
      setIsGenerating(false);
      onError?.('Generation stopped by user');
      return;
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    setIsGenerating(false);
  }, [onError]);

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
