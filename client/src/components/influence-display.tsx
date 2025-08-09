import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Brain, History, Sparkles } from 'lucide-react';

interface InfluenceDisplayProps {
  currentInfluence: string;
  influenceHistory: Array<{ time: string; text: string; }>;
}

export function InfluenceDisplay({ currentInfluence, influenceHistory }: InfluenceDisplayProps) {
  const [displayText, setDisplayText] = useState('');
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (currentInfluence && currentInfluence !== displayText) {
      setIsAnimating(true);
      
      // Fade out current text, then fade in new text
      setTimeout(() => {
        setDisplayText(currentInfluence);
        setIsAnimating(false);
      }, 150);
    }
  }, [currentInfluence, displayText]);

  return (
    <div className="space-y-6">
      {/* Real-time Influence */}
      <Card className="bg-card border-border" data-testid="current-influence">
        <CardHeader>
          <CardTitle className="text-lg font-semibold flex items-center text-foreground">
            <Brain className="text-quantum-purple mr-2 h-5 w-5" />
            Quantum Influence
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="min-h-[60px] flex items-center">
            {displayText ? (
              <div 
                className={`text-sm text-quantum-purple italic bg-background rounded-lg p-3 border border-quantum-purple/20 transition-opacity duration-300 ${isAnimating ? 'opacity-50' : 'opacity-100'}`}
                data-testid="influence-text"
              >
                <Sparkles className="inline w-4 h-4 mr-2 text-quantum-cyan" />
                "{displayText}"
              </div>
            ) : (
              <div className="text-sm text-muted-foreground italic" data-testid="influence-waiting">
                Waiting for generation...
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Influence History */}
      <Card className="bg-card border-border" data-testid="influence-history">
        <CardHeader>
          <CardTitle className="text-lg font-semibold flex items-center text-foreground">
            <History className="text-quantum-indigo mr-2 h-5 w-5" />
            Recent Influences
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-48 overflow-y-auto" data-testid="influence-history-list">
            {influenceHistory.length > 0 ? (
              influenceHistory.map((item, index) => (
                <div 
                  key={index}
                  className="text-xs text-muted-foreground bg-background rounded p-2 border border-border transition-all duration-300 hover:border-quantum-purple/30"
                  data-testid={`influence-history-item-${index}`}
                >
                  <div className="text-quantum-cyan text-xs mb-1 font-mono">
                    {item.time}
                  </div>
                  <div className="italic">
                    "{item.text}"
                  </div>
                </div>
              ))
            ) : (
              <div className="text-sm text-muted-foreground text-center py-4" data-testid="no-influence-history">
                No influences recorded yet
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
