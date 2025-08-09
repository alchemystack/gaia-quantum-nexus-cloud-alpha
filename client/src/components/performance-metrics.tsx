import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Zap, Database, Wifi } from 'lucide-react';
import type { PerformanceMetrics, QRNGStatus } from '../types/quantum';

interface PerformanceMetricsProps {
  metrics: PerformanceMetrics;
  qrngStatus: QRNGStatus | null;
}

export function PerformanceMetricsDisplay({ metrics, qrngStatus }: PerformanceMetricsProps) {
  const [displayMetrics, setDisplayMetrics] = useState(metrics);

  useEffect(() => {
    // Smooth animation of metric updates
    const timer = setTimeout(() => {
      setDisplayMetrics(metrics);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [metrics]);

  const getStatusColor = (status: boolean) => {
    return status ? 'text-quantum-green' : 'text-red-500';
  };

  const getLatencyColor = (latency: number) => {
    if (latency < 100) return 'text-quantum-green';
    if (latency < 200) return 'text-quantum-cyan';
    if (latency < 300) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getTokenSpeedColor = (speed: number) => {
    if (speed > 10) return 'text-quantum-green';
    if (speed > 5) return 'text-quantum-cyan';
    if (speed > 2) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <Card className="bg-card border-border" data-testid="performance-metrics">
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center text-foreground">
          <Activity className="text-quantum-green mr-2 h-5 w-5" />
          Performance Metrics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex justify-between items-center" data-testid="metric-latency">
          <span className="text-sm text-muted-foreground">Latency</span>
          <span className={`text-sm font-mono transition-colors ${getLatencyColor(displayMetrics.latency)}`}>
            {displayMetrics.latency || '--'} ms
          </span>
        </div>
        
        <div className="flex justify-between items-center" data-testid="metric-tokens-per-sec">
          <span className="text-sm text-muted-foreground">Tokens/sec</span>
          <span className={`text-sm font-mono transition-colors ${getTokenSpeedColor(displayMetrics.tokensPerSec)}`}>
            <Zap className="inline w-3 h-3 mr-1" />
            {displayMetrics.tokensPerSec || '--'} t/s
          </span>
        </div>
        
        <div className="flex justify-between items-center" data-testid="metric-entropy-used">
          <span className="text-sm text-muted-foreground">Entropy Used</span>
          <span className="text-sm font-mono text-quantum-purple">
            <Database className="inline w-3 h-3 mr-1" />
            {displayMetrics.entropyUsed || '--'} bits
          </span>
        </div>
        
        <div className="flex justify-between items-center" data-testid="metric-qrng-status">
          <span className="text-sm text-muted-foreground">QRNG Status</span>
          <div className="flex items-center">
            <Wifi className={`w-3 h-3 mr-1 ${getStatusColor(qrngStatus?.available || false)}`} />
            <span className={`text-sm font-mono ${getStatusColor(qrngStatus?.available || false)}`}>
              {qrngStatus?.available ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
        
        {qrngStatus?.provider && (
          <div className="text-xs text-muted-foreground pt-2 border-t border-border" data-testid="qrng-provider">
            Provider: {qrngStatus.provider}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
