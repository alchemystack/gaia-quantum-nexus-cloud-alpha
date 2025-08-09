import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Atom, Play, Square, Download, Edit, Terminal } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

import { QuantumParticles } from '../components/quantum-particles';
import { PerformanceMetricsDisplay } from '../components/performance-metrics';
import { EntropyPool } from '../components/entropy-pool';
import { InfluenceDisplay } from '../components/influence-display';
import { LayerAnalysisDisplay } from '../components/layer-analysis';
import { useWebSocket } from '../hooks/use-websocket';

import type { 
  GenerationRequest, 
  TokenResponse, 
  PerformanceMetrics, 
  LayerAnalysis, 
  QRNGStatus
} from '../types/quantum';

const QUANTUM_PROFILES = [
  { value: 'strict', label: 'Strict (Deterministic)', description: 'Linear deterministic processing' },
  { value: 'light', label: 'Light Quantum', description: 'Subtle quantum coherence patterns' },
  { value: 'medium', label: 'Medium Quantum', description: 'Dynamic consciousness integration' },
  { value: 'spicy', label: 'Spicy Quantum', description: 'Full quantum-karmic resonance' }
] as const;

export default function QuantumInterface() {
  // UI State
  const [prompt, setPrompt] = useState('');
  const [profile, setProfile] = useState<'strict' | 'light' | 'medium' | 'spicy'>('medium');
  const [temperature, setTemperature] = useState([0.7]);
  const [maxTokens, setMaxTokens] = useState(128);
  
  // Generation State  
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);
  const [vectorInterpretationTokens, setVectorInterpretationTokens] = useState<string[]>([]); // Vector interpretation output
  const [tokenCount, setTokenCount] = useState(0);
  const [currentInfluence, setCurrentInfluence] = useState('');
  const [influenceHistory, setInfluenceHistory] = useState<Array<{ time: string; text: string }>>([]);
  const [layerAnalysis, setLayerAnalysis] = useState<LayerAnalysis>({ attention: 0, ffn: 0, embedding: 0 });
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({ latency: 0, tokensPerSec: 0, entropyUsed: 0 });
  const [qrngStatus, setQrngStatus] = useState<QRNGStatus | null>(null);

  // Refs
  const outputRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLSpanElement>(null);
  
  const { toast } = useToast();

  // WebSocket connection
  const {
    isConnected,
    isConnecting,
    isGenerating,
    sendMessage,
    stopGeneration,
    reconnect
  } = useWebSocket({
    url: '/ws/generate',
    onToken: (response: TokenResponse) => {
      setGeneratedTokens(prev => [...prev, response.token]);
      
      // Parse vector interpretation from influence string
      // Format: "QRNG Vector: [...] → interpretation"
      if (response.influence.includes('→')) {
        const parts = response.influence.split('→');
        if (parts.length > 1) {
          const interpretation = parts[1].trim();
          setVectorInterpretationTokens(prev => [...prev, interpretation]);
        }
      } else {
        setVectorInterpretationTokens(prev => [...prev, response.token]); // Fallback to same token
      }
      
      setTokenCount(prev => prev + 1);
      setCurrentInfluence(response.influence);
      setLayerAnalysis(response.layerAnalysis);
      setPerformanceMetrics(response.performanceMetrics);
      
      // Add to influence history
      const now = new Date().toLocaleTimeString();
      setInfluenceHistory(prev => [
        { time: now, text: response.influence },
        ...prev.slice(0, 9) // Keep last 10 influences
      ]);
      
      // Scroll to bottom
      if (outputRef.current) {
        outputRef.current.scrollTop = outputRef.current.scrollHeight;
      }
    },
    onComplete: (sessionId: string, totalTokens: number) => {
      if (cursorRef.current) {
        cursorRef.current.style.opacity = '0';
      }
      toast({
        title: 'Generation Complete',
        description: `Generated ${totalTokens} tokens with quantum consciousness influence.`,
      });
    },
    onError: (error: string) => {
      // Check if this is a QRNG failure - show critical warning
      if (error.toLowerCase().includes('qrng') || error.toLowerCase().includes('quantum')) {
        toast({
          title: '⚠️ QUANTUM RANDOMNESS REQUIRED',
          description: 'Generation HALTED - This system requires TRUE quantum randomness. NO pseudorandom fallback exists. The QRNG API must be accessible for generation to proceed.',
          variant: 'destructive',
          duration: 10000, // Show for 10 seconds
        });
      } else {
        toast({
          title: 'Generation Error',
          description: error,
          variant: 'destructive'
        });
      }
    }
  });

  // Fetch QRNG status
  useEffect(() => {
    const fetchQRNGStatus = async () => {
      try {
        const response = await fetch('/api/qrng-status');
        if (response.ok) {
          const status = await response.json();
          setQrngStatus(status);
        }
      } catch (error) {
        console.error('Failed to fetch QRNG status:', error);
      }
    };

    fetchQRNGStatus();
    // Poll status every 10 seconds
    const interval = setInterval(fetchQRNGStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // Set initial prompt
  useEffect(() => {
    setPrompt("Explore the intersection of quantum consciousness and artificial intelligence, revealing how uncertainty principles might guide the emergence of truly creative AI systems.");
  }, []);

  // Show cursor while generating
  useEffect(() => {
    if (isGenerating && cursorRef.current) {
      cursorRef.current.style.opacity = '1';
    }
  }, [isGenerating]);

  const handleGenerate = () => {
    if (!prompt.trim() || isGenerating) return;

    // Check if quantum profile is selected but QRNG is unavailable
    if (profile !== 'strict' && !qrngStatus?.available) {
      toast({
        title: 'Quantum Randomness Required',
        description: 'This consciousness level requires true quantum randomness, but the QRNG service is offline. Please select "Strict (Deterministic)" mode or wait for QRNG to come online.',
        variant: 'destructive'
      });
      return;
    }

    // Reset state for both outputs
    setGeneratedTokens([]);
    setVectorInterpretationTokens([]); // Reset vector interpretation
    setTokenCount(0);
    setCurrentInfluence('');
    setInfluenceHistory([]);
    setLayerAnalysis({ attention: 0, ffn: 0, embedding: 0 });
    setPerformanceMetrics({ latency: 0, tokensPerSec: 0, entropyUsed: 0 });

    const request: GenerationRequest = {
      prompt: prompt.trim(),
      profile,
      maxTokens,
      temperature: temperature[0]
    };

    const success = sendMessage(request);
    if (!success) {
      toast({
        title: 'Connection Error',
        description: 'Unable to send generation request. Please check connection.',
        variant: 'destructive'
      });
    }
  };

  const handleStop = () => {
    stopGeneration();
    if (cursorRef.current) {
      cursorRef.current.style.opacity = '0';
    }
  };

  const handleExport = () => {
    if (generatedTokens.length === 0) {
      toast({
        title: 'No Content to Export',
        description: 'Generate some text first before exporting.',
        variant: 'destructive'
      });
      return;
    }

    const content = {
      prompt,
      profile,
      generatedText: generatedTokens.join(' '),
      quantumInfluences: influenceHistory,
      performanceMetrics,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gaia-quantum-session-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: 'Export Successful',
      description: 'Quantum session data exported to file.',
    });
  };

  const getConnectionStatus = () => {
    if (isConnecting) return { text: 'Connecting...', color: 'bg-yellow-500 animate-pulse' };
    if (isConnected) return { text: 'Connected', color: 'bg-quantum-green' };
    return { text: 'Disconnected', color: 'bg-red-500' };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <QuantumParticles />
      
      {/* Header */}
      <header className="relative z-10 border-b border-border bg-background/90 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-quantum-indigo to-quantum-purple rounded-lg flex items-center justify-center animate-quantum-glow" data-testid="logo">
                <Atom className="text-white text-lg h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold gradient-text" data-testid="app-title">
                  Gaia Quantum Nexus
                </h1>
                <p className="text-sm text-muted-foreground">Quantum-Augmented Large Language Model</p>
                <p className="text-xs text-yellow-500/90 font-mono">⚠️ TRUE QUANTUM ONLY - NO PSEUDORANDOM FALLBACK</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-muted-foreground flex items-center" data-testid="model-engine-status">
                <div className={`w-2 h-2 rounded-full mr-2 ${(qrngStatus as any)?.modelEngine === 'Modal GPT-OSS 120B' ? 'bg-purple-500 animate-pulse' : 'bg-yellow-500'}`} />
                {(qrngStatus as any)?.modelEngine || 'Unknown Model'}
              </div>
              <div className="text-sm text-muted-foreground flex items-center" data-testid="qrng-status-header">
                <div className={`w-2 h-2 rounded-full mr-2 ${qrngStatus?.available ? 'bg-quantum-green animate-pulse' : 'bg-red-500'}`} />
                QRNG {qrngStatus?.available ? 'Active' : 'Offline'}
              </div>
              <Button 
                onClick={handleExport} 
                variant="outline"
                className="border-quantum-indigo text-quantum-indigo hover:bg-quantum-indigo hover:text-white"
                data-testid="button-export"
              >
                <Download className="mr-2 h-4 w-4" />
                Export Session
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Left Panel: Input Controls */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Quantum Influence Profile */}
            <Card className="bg-card border-border" data-testid="controls-card">
              <CardHeader>
                <CardTitle className="text-lg font-semibold flex items-center text-foreground">
                  <Edit className="text-quantum-cyan mr-2 h-5 w-5" />
                  Quantum Influence
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="profile" className="text-sm font-medium text-foreground">Consciousness Level</Label>
                  <Select value={profile} onValueChange={(value: 'strict' | 'light' | 'medium' | 'spicy') => setProfile(value)}>
                    <SelectTrigger className="w-full mt-2" data-testid="select-profile">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {QUANTUM_PROFILES.map((p) => (
                        <SelectItem 
                          key={p.value} 
                          value={p.value} 
                          data-testid={`profile-option-${p.value}`}
                          disabled={p.value !== 'strict' && !qrngStatus?.available}
                        >
                          <div className="flex items-center justify-between w-full">
                            <span className={p.value !== 'strict' && !qrngStatus?.available ? 'text-muted-foreground' : ''}>
                              {p.label}
                            </span>
                            {p.value !== 'strict' && !qrngStatus?.available && (
                              <span className="text-xs text-red-500 ml-2">QRNG Required</span>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div className="mt-1 space-y-1">
                    <p className="text-xs text-muted-foreground" data-testid="profile-description">
                      {QUANTUM_PROFILES.find(p => p.value === profile)?.description}
                    </p>
                    {profile !== 'strict' && !qrngStatus?.available && (
                      <p className="text-xs text-red-500" data-testid="qrng-warning">
                        ⚠️ Quantum randomness unavailable - true QRNG required for this mode
                      </p>
                    )}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label htmlFor="temperature" className="text-sm font-medium text-foreground">Temperature</Label>
                    <Slider
                      value={temperature}
                      onValueChange={setTemperature}
                      max={2}
                      min={0.1}
                      step={0.1}
                      className="mt-2"
                      data-testid="slider-temperature"
                    />
                    <span className="text-xs text-muted-foreground" data-testid="temperature-value">{temperature[0]}</span>
                  </div>
                  <div>
                    <Label htmlFor="max-tokens" className="text-sm font-medium text-foreground">Max Tokens</Label>
                    <Input
                      type="number"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      min={1}
                      max={1000}
                      className="mt-2"
                      data-testid="input-max-tokens"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <PerformanceMetricsDisplay metrics={performanceMetrics} qrngStatus={qrngStatus} />

            {/* Entropy Pool */}
            <EntropyPool qrngStatus={qrngStatus} />
          </div>

          {/* Center Panel: Text Generation */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Prompt Input */}
            <Card className="bg-card border-border" data-testid="prompt-card">
              <CardHeader>
                <CardTitle className="text-lg font-semibold flex items-center text-foreground">
                  <Edit className="text-quantum-indigo mr-2 h-5 w-5" />
                  Prompt Input
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  placeholder="Enter your prompt to explore quantum-influenced text generation..."
                  className="resize-none"
                  disabled={isGenerating}
                  data-testid="textarea-prompt"
                />
                <div className="flex justify-between items-center">
                  <div className="flex space-x-2">
                    <Button 
                      onClick={handleGenerate}
                      disabled={!prompt.trim() || isGenerating || (profile !== 'strict' && !qrngStatus?.available)}
                      className="bg-gradient-to-r from-quantum-indigo to-quantum-purple hover:from-quantum-purple hover:to-quantum-indigo disabled:from-gray-600 disabled:to-gray-600"
                      data-testid="button-generate"
                    >
                      <Play className="mr-2 h-4 w-4" />
                      {isGenerating ? 'Generating...' : profile !== 'strict' && !qrngStatus?.available ? 'Quantum Randomness Required' : 'Generate'}
                    </Button>
                    <Button 
                      onClick={handleStop}
                      disabled={!isGenerating}
                      variant="outline"
                      data-testid="button-stop"
                    >
                      <Square className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="text-sm text-muted-foreground" data-testid="char-count">
                    {prompt.length} characters
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Dual Output Display */}
            <Card className="bg-card border-border" data-testid="output-card">
              <CardHeader>
                <CardTitle className="text-lg font-semibold flex items-center justify-between text-foreground">
                  <span className="flex items-center">
                    <Terminal className="text-quantum-cyan mr-2 h-5 w-5" />
                    Dual Output Display
                  </span>
                  <div className="flex items-center space-x-2 text-sm">
                    <span className="text-muted-foreground">Tokens:</span>
                    <span className="font-mono text-quantum-green" data-testid="token-count">{tokenCount}</span>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* QRNG-Modified Output */}
                  <div>
                    <h4 className="text-sm font-semibold text-quantum-indigo mb-2">QRNG-Modified Output</h4>
                    <div 
                      ref={outputRef}
                      className="bg-background rounded-lg p-4 min-h-[250px] max-h-[350px] overflow-y-auto border border-border font-mono text-sm leading-relaxed"
                      data-testid="output-container"
                    >
                      {prompt && (
                        <div className="text-muted-foreground mb-2" data-testid="prompt-echo">
                          {prompt}
                        </div>
                      )}
                      <div className="text-foreground" data-testid="generated-text">
                        {generatedTokens.join(' ')}
                      </div>
                      <span 
                        ref={cursorRef}
                        className="inline-block w-2 h-5 bg-quantum-indigo animate-pulse ml-1 opacity-0"
                        data-testid="cursor"
                      />
                    </div>
                  </div>
                  
                  {/* Vector Interpretation Output */}
                  <div>
                    <h4 className="text-sm font-semibold text-quantum-purple mb-2">Vector Interpretation (1s-scaled)</h4>
                    <div 
                      className="bg-background rounded-lg p-4 min-h-[250px] max-h-[350px] overflow-y-auto border border-border font-mono text-sm leading-relaxed"
                      data-testid="vector-interpretation-container"
                    >
                      {prompt && (
                        <div className="text-muted-foreground mb-2">
                          {prompt}
                        </div>
                      )}
                      <div className="text-foreground" data-testid="vector-interpretation-text">
                        {vectorInterpretationTokens.join(' ')}
                      </div>
                    </div>
                  </div>
                </div>
                

              </CardContent>
            </Card>
          </div>

          {/* Right Panel: Quantum Influence Translator */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Real-time Influence */}
            <InfluenceDisplay 
              currentInfluence={currentInfluence}
              influenceHistory={influenceHistory}
            />

            {/* Layer Analysis */}
            <LayerAnalysisDisplay analysis={layerAnalysis} />
          </div>
        </div>
      </main>

      {/* WebSocket Status Indicator */}
      <div className="fixed bottom-4 left-4 px-3 py-2 bg-card border border-border rounded-lg text-sm flex items-center space-x-2" data-testid="ws-status">
        <div className={`w-2 h-2 rounded-full ${connectionStatus.color}`} />
        <span>{connectionStatus.text}</span>
        {!isConnected && (
          <Button 
            onClick={reconnect}
            size="sm"
            variant="outline"
            className="ml-2"
            data-testid="button-reconnect"
          >
            Retry
          </Button>
        )}
      </div>
    </div>
  );
}
