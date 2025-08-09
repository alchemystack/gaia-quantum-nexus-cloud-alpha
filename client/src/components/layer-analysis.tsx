import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Layers } from 'lucide-react';
import type { LayerAnalysis } from '../types/quantum';

interface LayerAnalysisProps {
  analysis: LayerAnalysis;
}

interface AnimatedLayer {
  name: string;
  value: number;
  displayValue: number;
  color: string;
  icon: string;
}

export function LayerAnalysisDisplay({ analysis }: LayerAnalysisProps) {
  const [layers, setLayers] = useState<AnimatedLayer[]>([
    { name: 'Attention', value: 0, displayValue: 0, color: 'quantum-cyan', icon: '🎯' },
    { name: 'FFN', value: 0, displayValue: 0, color: 'quantum-purple', icon: '⚡' },
    { name: 'Embedding', value: 0, displayValue: 0, color: 'quantum-green', icon: '🌟' }
  ]);

  useEffect(() => {
    const newLayers = layers.map(layer => ({
      ...layer,
      value: (analysis[layer.name.toLowerCase()] || 0) * 100
    }));
    
    setLayers(newLayers);

    // Animate the display values
    const timers = newLayers.map((layer, index) => 
      setTimeout(() => {
        setLayers(prev => prev.map((l, i) => 
          i === index ? { ...l, displayValue: layer.value } : l
        ));
      }, index * 100)
    );

    return () => timers.forEach(clearTimeout);
  }, [analysis]);

  const getIntensityLabel = (value: number) => {
    if (value < 20) return 'Low';
    if (value < 50) return 'Medium';
    if (value < 80) return 'High';
    return 'Peak';
  };

  const getProgressColor = (color: string, value: number) => {
    const intensity = value / 100;
    const colors = {
      'quantum-cyan': `hsl(186, 77%, ${47 + intensity * 20}%)`,
      'quantum-purple': `hsl(264, 83%, ${70 + intensity * 10}%)`,
      'quantum-green': `hsl(158, 64%, ${52 + intensity * 15}%)`
    };
    return colors[color as keyof typeof colors] || colors['quantum-cyan'];
  };

  return (
    <Card className="bg-card border-border" data-testid="layer-analysis">
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center text-foreground">
          <Layers className="text-quantum-green mr-2 h-5 w-5" />
          Layer Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {layers.map((layer, index) => (
            <div key={layer.name} className="space-y-2" data-testid={`layer-${layer.name.toLowerCase()}`}>
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center">
                  <span className="mr-2">{layer.icon}</span>
                  <span className="text-foreground font-medium">{layer.name}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`text-xs px-2 py-1 rounded-full bg-${layer.color}/20 text-${layer.color} font-mono`}>
                    {getIntensityLabel(layer.displayValue)}
                  </span>
                  <span className={`text-sm font-mono text-${layer.color}`} data-testid={`layer-${layer.name.toLowerCase()}-value`}>
                    {Math.round(layer.displayValue)}%
                  </span>
                </div>
              </div>
              
              <div className="relative">
                <Progress 
                  value={layer.displayValue} 
                  className="h-2 bg-muted"
                  data-testid={`layer-${layer.name.toLowerCase()}-progress`}
                />
                <div 
                  className="absolute inset-0 h-2 rounded-full transition-all duration-500 opacity-80"
                  style={{ 
                    width: `${layer.displayValue}%`,
                    backgroundColor: getProgressColor(layer.color, layer.displayValue),
                    boxShadow: `0 0 10px ${getProgressColor(layer.color, layer.displayValue)}`
                  }}
                />
              </div>
            </div>
          ))}
          
          <div className="pt-2 border-t border-border">
            <div className="text-xs text-muted-foreground">
              Quantum influence distribution across neural layers
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
