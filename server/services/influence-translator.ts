export class InfluenceTranslator {
  // Profile descriptions without arbitrary quantum words
  private profileDescriptions = {
    strict: "Deterministic mode - no QRNG modification",
    light: "QRNG modifier strength: 0.3",
    medium: "QRNG modifier strength: 0.6", 
    spicy: "QRNG modifier strength: 1.0"
  };

  translateInfluence(profile: string, layerAnalysis: Record<string, number>): string {
    if (profile === 'strict') {
      return this.profileDescriptions.strict;
    }

    // Return technical information about the modification
    const activeValues = Object.entries(layerAnalysis)
      .map(([layer, value]) => `${layer}: ${value.toFixed(3)}`)
      .join(', ');
    
    return `Layer values: ${activeValues}`;
  }

  getProfileDescription(profile: string): string {
    return this.profileDescriptions[profile as keyof typeof this.profileDescriptions] 
      || "Unknown profile";
  }

  // Return technical data instead of arbitrary phrases
  generateTechnicalMetric(entropyUsed: number): string {
    return `Entropy consumed: ${entropyUsed} bytes`;
  }
}

export const influenceTranslator = new InfluenceTranslator();
