export interface QuantumInfluenceProfile {
  name: string;
  value: 'strict' | 'light' | 'medium' | 'spicy';
  description: string;
  noiseLevel: number;
  color: string;
}

export interface PerformanceMetrics {
  latency: number;
  tokensPerSec: number;
  entropyUsed: number;
}

export interface LayerAnalysis {
  attention: number;
  ffn: number;
  embedding: number;
  [key: string]: number;
}

export interface TokenResponse {
  token: string;
  influence: string;
  layerAnalysis: LayerAnalysis;
  performanceMetrics: PerformanceMetrics;
}

export interface GenerationRequest {
  prompt: string;
  profile: 'strict' | 'light' | 'medium' | 'spicy';
  maxTokens: number;
  temperature: number;
}

export interface QRNGStatus {
  available: boolean;
  provider: string;
  entropyPool: {
    size: number;
    percentage: number;
  };
}

export interface WebSocketMessage {
  type?: 'token' | 'complete' | 'error';
  token?: string;
  influence?: string;
  layerAnalysis?: LayerAnalysis;
  performanceMetrics?: PerformanceMetrics;
  sessionId?: string;
  totalTokens?: number;
  message?: string;
}

export const QUANTUM_PROFILES: QuantumInfluenceProfile[] = [
  {
    name: 'Strict (Deterministic)',
    value: 'strict',
    description: 'Linear deterministic processing',
    noiseLevel: 0,
    color: 'text-gray-400'
  },
  {
    name: 'Light Quantum',
    value: 'light',
    description: 'Subtle quantum coherence patterns',
    noiseLevel: 0.2,
    color: 'text-quantum-cyan'
  },
  {
    name: 'Medium Quantum',
    value: 'medium',
    description: 'Dynamic consciousness integration',
    noiseLevel: 0.5,
    color: 'text-quantum-purple'
  },
  {
    name: 'Spicy Quantum',
    value: 'spicy', 
    description: 'Full quantum-karmic resonance',
    noiseLevel: 0.8,
    color: 'text-quantum-green'
  }
];
