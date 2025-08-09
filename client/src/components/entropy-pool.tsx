import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Database } from 'lucide-react';
import type { QRNGStatus } from '../types/quantum';

interface EntropyPoolProps {
  qrngStatus: QRNGStatus | null;
}

export function EntropyPool({ qrngStatus }: EntropyPoolProps) {
  const [displayPercentage, setDisplayPercentage] = useState(0);
  
  const percentage = qrngStatus?.entropyPool?.percentage || 0;

  useEffect(() => {
    // Smooth animation for percentage changes
    const timer = setTimeout(() => {
      setDisplayPercentage(percentage);
    }, 200);
    
    return () => clearTimeout(timer);
  }, [percentage]);

  const getPoolColor = (percent: number) => {
    if (percent > 70) return 'text-quantum-green';
    if (percent > 40) return 'text-quantum-cyan';
    if (percent > 20) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getProgressColor = (percent: number) => {
    if (percent > 70) return 'from-quantum-green to-quantum-cyan';
    if (percent > 40) return 'from-quantum-cyan to-quantum-purple';
    if (percent > 20) return 'from-yellow-400 to-quantum-cyan';
    return 'from-red-400 to-yellow-400';
  };

  return (
    <Card className="bg-card border-border" data-testid="entropy-pool">
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center text-foreground">
          <Database className="text-quantum-purple mr-2 h-5 w-5" />
          Entropy Pool
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="relative">
            <Progress 
              value={displayPercentage} 
              className="h-3 bg-muted"
              data-testid="entropy-progress-bar"
            />
            <div 
              className={`absolute inset-0 h-3 rounded-full bg-gradient-to-r ${getProgressColor(displayPercentage)} transition-all duration-300`}
              style={{ width: `${displayPercentage}%` }}
            />
          </div>
          
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Buffer</span>
            <span className={`font-mono transition-colors ${getPoolColor(displayPercentage)}`} data-testid="entropy-percentage">
              {Math.round(displayPercentage)}%
            </span>
          </div>
          
          <div className="text-xs text-muted-foreground mt-2" data-testid="entropy-source">
            {qrngStatus?.available ? (
              <>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-quantum-green rounded-full mr-2 animate-pulse" />
                  ID Quantique QUANTIS source active
                </div>
                {qrngStatus.entropyPool?.size && (
                  <div className="mt-1 text-quantum-cyan">
                    Pool size: {qrngStatus.entropyPool.size} bytes
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center">
                <div className="w-2 h-2 bg-red-500 rounded-full mr-2" />
                Fallback to cryptographic randomness
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
