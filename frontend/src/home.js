import React, { useState } from 'react';
import { Camera, Mic, Heart, Brain, Stethoscope, Activity, CheckCircle, ArrowRight, Play, Info, Zap, Award, Shield } from 'lucide-react';

export default function EchoesLandingPage({onComplete}) {
  const [currentStep, setCurrentStep] = useState(0);
  const [showApp, setShowApp] = useState(false);
  const [userName, setUserName] = useState('');
  const [userRole, setUserRole] = useState('');

  const steps = [
    {
      title: "Welcome to ECHOES",
      subtitle: "Your AI-Powered Speech Training Assistant",
      icon: Stethoscope,
      color: "from-blue-500 to-cyan-500",
      content: (
        <div className="space-y-6">
          <div className="text-center">
            <div className="inline-block bg-gradient-to-r from-blue-500 to-cyan-500 p-4 rounded-full mb-4">
              <Stethoscope size={48} className="text-white" />
            </div>
            <h2 className="text-4xl font-bold text-white mb-4">ECHOES</h2>
            <p className="text-xl text-gray-300 mb-6">
              Advanced Lip Reading Technology for Speech Rehabilitation
            </p>
          </div>

          <div className="grid sm:grid-cols-3 gap-4">
            <div className="bg-white/10 rounded-xl p-6 border border-white/20 text-center hover:bg-white/20 transition-all transform hover:scale-105">
              <Camera size={32} className="text-blue-400 mx-auto mb-3" />
              <h3 className="font-semibold text-white mb-2">Real-Time Tracking</h3>
              <p className="text-sm text-gray-300">Advanced AI detects lip movements instantly</p>
            </div>
            <div className="bg-white/10 rounded-xl p-6 border border-white/20 text-center hover:bg-white/20 transition-all transform hover:scale-105">
              <Brain size={32} className="text-purple-400 mx-auto mb-3" />
              <h3 className="font-semibold text-white mb-2">Emotion Detection</h3>
              <p className="text-sm text-gray-300">ML-powered facial expression analysis</p>
            </div>
            <div className="bg-white/10 rounded-xl p-6 border border-white/20 text-center hover:bg-white/20 transition-all transform hover:scale-105">
              <Activity size={32} className="text-green-400 mx-auto mb-3" />
              <h3 className="font-semibold text-white mb-2">Progress Reports</h3>
              <p className="text-sm text-gray-300">Detailed analytics and tracking</p>
            </div>
          </div>

          <div className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/30 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <Info className="text-yellow-400 flex-shrink-0 mt-1" size={20} />
              <div>
                <p className="text-white font-medium mb-1">Who is this for?</p>
                <p className="text-sm text-gray-300">
                  Healthcare providers working with speech rehabilitation patients, therapists, 
                  and individuals recovering from speech impairments.
                </p>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "About You",
      subtitle: "Help us personalize your experience",
      icon: Heart,
      color: "from-violet-500 to-purple-600",
      content: (
        <div className="space-y-6">
          <div className="text-center mb-6">
            <div className="inline-block bg-gradient-to-r from-purple-500 to-pink-500 p-4 rounded-full mb-4">
              <Heart size={48} className="text-white" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">Tell us about yourself</h3>
            <p className="text-gray-300">This helps us customize your experience</p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-white font-medium mb-2">Your Name (Optional)</label>
              <input
                type="text"
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
                placeholder="Enter your name"
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            <div>
              <label className="block text-white font-medium mb-3">I am a...</label>
              <div className="grid sm:grid-cols-2 gap-3">
                {['Healthcare Provider', 'Patient', 'Therapist', 'Caregiver'].map((role) => (
                  <button
                    key={role}
                    onClick={() => setUserRole(role)}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      userRole === role
                        ? 'bg-purple-500 border-purple-400 text-white'
                        : 'bg-white/10 border-white/20 text-gray-300 hover:bg-white/20'
                    }`}
                  >
                    <p className="font-medium">{role}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg p-4">
            <p className="text-sm text-blue-200">
              💡 <strong>Privacy Note:</strong> All information stays on your device. 
              We don't store or transmit personal data.
            </p>
          </div>
        </div>
      )
    },
    {
      title: "How It Works",
      subtitle: "Simple 3-step process",
      icon: Zap,
      color: "from-green-500 to-emerald-500",
      content: (
        <div className="space-y-6">
          <div className="text-center mb-6">
            <div className="inline-block bg-gradient-to-r from-green-500 to-emerald-500 p-4 rounded-full mb-4">
              <Zap size={48} className="text-white" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">How ECHOES Works</h3>
            <p className="text-gray-300">Follow these simple steps to get started</p>
          </div>

          <div className="space-y-4">
            <div className="bg-white/10 rounded-xl p-5 border border-white/20 hover:bg-white/15 transition-all">
              <div className="flex items-start gap-4">
                <div className="bg-blue-500 rounded-full p-3 flex-shrink-0">
                  <Camera size={24} className="text-white" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="bg-blue-500 text-white text-sm font-bold px-2 py-1 rounded">
                      Step 1
                    </span>
                    <h4 className="text-white font-semibold">Position Your Camera</h4>
                  </div>
                  <p className="text-gray-300 text-sm">Ensure your face is well-lit and centered in the camera view</p>
                </div>
              </div>
            </div>

            <div className="bg-white/10 rounded-xl p-5 border border-white/20 hover:bg-white/15 transition-all">
              <div className="flex items-start gap-4">
                <div className="bg-purple-500 rounded-full p-3 flex-shrink-0">
                  <Mic size={24} className="text-white" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="bg-purple-500 text-white text-sm font-bold px-2 py-1 rounded">
                      Step 2
                    </span>
                    <h4 className="text-white font-semibold">Speak Clearly</h4>
                  </div>
                  <p className="text-gray-300 text-sm">Exaggerate lip movements while speaking slowly and clearly</p>
                </div>
              </div>
            </div>

            <div className="bg-white/10 rounded-xl p-5 border border-white/20 hover:bg-white/15 transition-all">
              <div className="flex items-start gap-4">
                <div className="bg-green-500 rounded-full p-3 flex-shrink-0">
                  <Activity size={24} className="text-white" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="bg-green-500 text-white text-sm font-bold px-2 py-1 rounded">
                      Step 3
                    </span>
                    <h4 className="text-white font-semibold">Get Real-Time Feedback</h4>
                  </div>
                  <p className="text-gray-300 text-sm">See your speech converted to text with emotion tracking</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-500/20 border border-yellow-400/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Shield className="text-yellow-400 flex-shrink-0 mt-1" size={20} />
              <div>
                <p className="text-white font-medium mb-1">Privacy & Security</p>
                <p className="text-sm text-gray-300">
                  All processing happens in real-time. Video is never stored or transmitted 
                  beyond your session.
                </p>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "System Requirements",
      subtitle: "Make sure you're ready",
      icon: CheckCircle,
      color: "from-cyan-500 to-blue-500",
      content: (
        <div className="space-y-6">
          <div className="text-center mb-6">
            <div className="inline-block bg-gradient-to-r from-cyan-500 to-blue-500 p-4 rounded-full mb-4">
              <CheckCircle size={48} className="text-white" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">Before You Start</h3>
            <p className="text-gray-300">Ensure your setup meets these requirements</p>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            <div className="space-y-3">
              <h4 className="text-white font-semibold flex items-center gap-2 mb-3">
                <CheckCircle className="text-green-400" size={20} />
                Required
              </h4>
              {[
                'Working webcam (720p or higher)',
                'Modern web browser (Chrome/Edge/Safari)',
                'Stable internet connection',
                'Well-lit room'
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2 bg-white/5 rounded-lg p-3">
                  <CheckCircle className="text-green-400 flex-shrink-0 mt-0.5" size={16} />
                  <span className="text-gray-300 text-sm">{item}</span>
                </div>
              ))}
            </div>

            <div className="space-y-3">
              <h4 className="text-white font-semibold flex items-center gap-2 mb-3">
                <Award className="text-blue-400" size={20} />
                Recommended
              </h4>
              {[
                'Quiet environment',
                'Neutral background',
                'Camera at eye level',
                'Clear pronunciation'
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2 bg-white/5 rounded-lg p-3">
                  <Award className="text-blue-400 flex-shrink-0 mt-0.5" size={16} />
                  <span className="text-gray-300 text-sm">{item}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-400/30 rounded-xl p-5">
            <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Brain className="text-green-400" size={20} />
              AI Features Included
            </h4>
            <div className="grid sm:grid-cols-2 gap-3">
              {[
                'Template-based word matching',
                'Real-time emotion detection (ML)',
                'Mouth region tracking',
                'Progress analytics'
              ].map((feature, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-green-100 text-sm">{feature}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    }
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setShowApp(true);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    setShowApp(true);
  };

  if (showApp) {
    
    if (onComplete) {
      setTimeout(() => onComplete(), 1500);
    }
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-8">
        <div className="text-center">
          <div className="inline-block bg-gradient-to-r from-green-500 to-blue-600 p-4 rounded-full mb-6 animate-pulse">
            <CheckCircle size={64} className="text-white" />
          </div>
          <h2 className="text-4xl font-bold text-white mb-4">
            {userName ? `Welcome, ${userName}!` : 'Welcome!'}
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Starting ECHOES {userRole && `for ${userRole}s`}...
          </p>
          <div className="flex items-center justify-center gap-2 text-blue-300">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    );
  }

  const CurrentIcon = steps[currentStep].icon;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4 sm:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-gray-400">
              Step {currentStep + 1} of {steps.length}
            </span>
            <button
              onClick={handleSkip}
              className="text-sm text-gray-400 hover:text-white transition-colors underline"
            >
              Skip Tutorial
            </button>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Main Content Card */}
        <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/20 p-8 sm:p-12">
          {/* Header */}
          <div className="text-center mb-8">
            <div className={`inline-block bg-gradient-to-r ${steps[currentStep].color} p-4 rounded-2xl mb-4 animate-pulse`}>
              <CurrentIcon size={48} className="text-white" />
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">
              {steps[currentStep].title}
            </h1>
            <p className="text-gray-300 text-lg">
              {steps[currentStep].subtitle}
            </p>
          </div>

          {/* Step Content */}
          <div className="mb-8">
            {steps[currentStep].content}
          </div>

          {/* Navigation Buttons */}
          <div className="flex items-center justify-between gap-4">
            <button
              onClick={handlePrev}
              disabled={currentStep === 0}
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                currentStep === 0
                  ? 'bg-white/5 text-gray-500 cursor-not-allowed'
                  : 'bg-white/10 hover:bg-white/20 text-white border border-white/20'
              }`}
            >
              Previous
            </button>

            <div className="flex gap-2">
              {steps.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full transition-all ${
                    index === currentStep
                      ? 'bg-blue-500 w-8'
                      : index < currentStep
                      ? 'bg-green-500'
                      : 'bg-white/20'
                  }`}
                />
              ))}
            </div>

            <button
              onClick={handleNext}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-green-600 hover:from-blue-600 hover:to-green-700 rounded-lg font-medium text-white shadow-lg transition-all transform hover:scale-105"
            >
              {currentStep === steps.length - 1 ? (
                <>
                  Start Session
                  <Play size={20} />
                </>
              ) : (
                <>
                  Next
                  <ArrowRight size={20} />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-gray-400 text-sm">
            Powered by Advanced AI • Template Matching • ML Emotion Detection
          </p>
        </div>
      </div>
    </div>
  );
}

