import React, { useState, useEffect } from 'react';
import { 
  AreaChart, Area, 
  BarChart, Bar, 
  XAxis, YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, 
  AlertTriangle, 
  DollarSign,
  BarChart2,
  Info,
  X,
  Shield,
  PieChart
} from 'lucide-react';

const TooltipContent = ({ title, description, interpretation, example }) => (
  <div className="max-w-xs bg-gray-700 p-4 rounded-lg shadow-lg">
    <h4 className="font-bold text-white mb-2">{title}</h4>
    <p className="text-gray-300 text-sm mb-2">{description}</p>
    <p className="text-gray-300 text-sm mb-2">{interpretation}</p>
    <p className="text-gray-400 text-sm italic">Example: {example}</p>
  </div>
);

const BeginnersGuide = ({ isOpen, onClose, currentStep, onNextStep }) => {
  const steps = [
    {
      title: "Welcome to Your Trading Dashboard",
      content: "This dashboard helps you track your trading performance and manage risk. Let's walk through the key features.",
      target: null
    },
    {
      title: "Daily Performance",
      content: "Here you can see how your trades performed today, including profits/losses and win rate.",
      target: "daily-metrics"
    },
    {
      title: "Position Overview",
      content: "This chart shows your current positions and their sizes. Green bars are long positions, red are short.",
      target: "position-chart"
    },
    {
      title: "Risk Analysis",
      content: "Monitor your risk levels here. Pay attention to position sizes and exposure levels.",
      target: "risk-analysis"
    }
  ];

  if (!isOpen) return null;

  const currentStepData = steps[currentStep];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 p-6 rounded-lg max-w-md">
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-xl font-bold">{currentStepData.title}</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X size={20} />
          </button>
        </div>
        <p className="text-gray-300 mb-6">{currentStepData.content}</p>
        <div className="flex justify-between">
          <button
            onClick={() => currentStep > 0 && onNextStep(currentStep - 1)}
            className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50"
            disabled={currentStep === 0}
          >
            Previous
          </button>
          <button
            onClick={() => {
              if (currentStep === steps.length - 1) {
                onClose();
              } else {
                onNextStep(currentStep + 1);
              }
            }}
            className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500"
          >
            {currentStep === steps.length - 1 ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
};

const RiskBadge = ({ risk }) => {
  let color;
  if (risk < 2) color = 'bg-green-500/10 text-green-400';
  else if (risk < 3) color = 'bg-yellow-500/10 text-yellow-400';
  else color = 'bg-red-500/10 text-red-400';

  return (
    <span className={`px-2 py-1 rounded-full text-xs ${color}`}>
      {risk < 2 ? 'Low' : risk < 3 ? 'Medium' : 'High'}
    </span>
  );
};

const MetricCard = ({ title, value, icon, trend, trendValue, onInfoClick, tooltip }) => (
  <div className="bg-gray-800 rounded-lg p-6 relative">
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center gap-2">
        <h3 className="text-gray-400 text-sm">{title}</h3>
        <button
          onClick={onInfoClick}
          className="text-gray-400 hover:text-white"
        >
          <Info size={16} />
        </button>
      </div>
      {icon}
    </div>
    <div className="flex items-baseline">
      <p className="text-2xl font-bold">{value}</p>
      <span className={`ml-2 text-sm ${
        trend === 'up' ? 'text-green-400' : 'text-red-400'
      }`}>
        {trendValue}
      </span>
    </div>
    {tooltip && (
      <div className="absolute z-10 bottom-full left-0 mb-2">
        <TooltipContent {...tooltip} />
      </div>
    )}
  </div>
);

const PositionsTab = ({ data }) => (
  <div className="grid gap-6">
    {/* Positions Summary */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Total Positions</h3>
        <p className="text-2xl font-bold">{data.positions.length}</p>
      </div>
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Long Exposure</h3>
        <p className="text-2xl font-bold text-green-400">
          ${data.riskMetrics.exposures.long_exposure.toLocaleString()}
        </p>
      </div>
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Short Exposure</h3>
        <p className="text-2xl font-bold text-red-400">
          ${Math.abs(data.riskMetrics.exposures.short_exposure).toLocaleString()}
        </p>
      </div>
    </div>

    {/* Positions Table */}
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4">Open Positions</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="px-6 py-3 text-left text-sm font-medium text-gray-400">Symbol</th>
              <th className="px-6 py-3 text-right text-sm font-medium text-gray-400">Quantity</th>
              <th className="px-6 py-3 text-right text-sm font-medium text-gray-400">Value</th>
              <th className="px-6 py-3 text-right text-sm font-medium text-gray-400">Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {data.positions.map((position) => (
              <tr key={position.symbol} className="border-b border-gray-700">
                <td className="px-6 py-4 whitespace-nowrap text-sm">{position.symbol}</td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                  <span className={position.quantity > 0 ? 'text-green-400' : 'text-red-400'}>
                    {position.quantity}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                  ${Math.abs(position.value).toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <RiskBadge risk={position.risk} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </div>
);

const RiskAnalysisTab = ({ data }) => (
  <div className="grid gap-6">
    {/* Risk Metrics Summary */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Profit Factor</h3>
        <p className="text-2xl font-bold">{data.riskMetrics.profitFactor.toFixed(2)}</p>
      </div>
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Max Drawdown</h3>
        <p className="text-2xl font-bold text-red-400">
          ${Math.abs(data.riskMetrics.maxDrawdown).toLocaleString()}
        </p>
      </div>
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-gray-400 text-sm mb-2">Net Exposure</h3>
        <p className="text-2xl font-bold">
          ${data.riskMetrics.exposures.net_exposure.toLocaleString()}
        </p>
      </div>
    </div>

    {/* Risk Alerts */}
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <AlertTriangle className="h-6 w-6 text-yellow-400" />
        Risk Alerts
      </h2>
      <div className="space-y-4">
        {data.alerts.map((alert, index) => (
          <div key={index} className="bg-gray-700/50 rounded-lg p-4 flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <p className="text-gray-300">{alert}</p>
          </div>
        ))}
        {data.alerts.length === 0 && (
          <div className="bg-gray-700/50 rounded-lg p-4 flex items-start gap-3">
            <Shield className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
            <p className="text-gray-300">No risk alerts at this time.</p>
          </div>
        )}
      </div>
    </div>

    {/* Exposure Analysis */}
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4">Exposure Analysis</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-gray-400 text-sm mb-4">Gross vs Net Exposure</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Gross Exposure</span>
              <span className="text-white font-medium">
                ${data.riskMetrics.exposures.gross_exposure.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Net Exposure</span>
              <span className="text-white font-medium">
                ${data.riskMetrics.exposures.net_exposure.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Long Exposure</span>
              <span className="text-green-400 font-medium">
                ${data.riskMetrics.exposures.long_exposure.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Short Exposure</span>
              <span className="text-red-400 font-medium">
                ${Math.abs(data.riskMetrics.exposures.short_exposure).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
        
        <div>
          <h3 className="text-gray-400 text-sm mb-4">Sector Exposure</h3>
          <div className="space-y-3">
            {Object.entries(data.riskMetrics.exposures.sector_exposure || {}).map(([sector, exposure]) => (
              <div key={sector} className="flex justify-between items-center">
                <span className="text-gray-400">{sector}</span>
                <span className="text-white font-medium">
                  ${Math.abs(exposure).toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
);

const TradingDashboard = () => {
  const [data, setData] = useState(null);
  const [explanations, setExplanations] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showGuide, setShowGuide] = useState(false);
  const [guideStep, setGuideStep] = useState(0);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dashboardResponse, explanationsResponse] = await Promise.all([
          fetch('http://localhost:8000/api/dashboard/overview'),
          fetch('http://localhost:8000/api/dashboard/metrics/explanations')
        ]);

        const dashboardData = await dashboardResponse.json();
        const explanationsData = await explanationsResponse.json();

        setData(dashboardData);
        setExplanations(explanationsData);
        setIsLoading(false);
      } catch (err) {
        setError(err.message);
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'positions':
        return <PositionsTab data={data} />;
      case 'risk':
        return <RiskAnalysisTab data={data} />;
      default:
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* P&L Chart */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4">Daily P&L</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.pnlHistory}>
                    <defs>
                      <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#34D399" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#34D399" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="date" 
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                      tickFormatter={(value) => `$${value}`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937',
                        border: 'none',
                        borderRadius: '0.375rem',
                        color: '#F3F4F6'
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#34D399"
                      fill="url(#profitGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Position Size Chart */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4">Position Sizes</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.positions}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="symbol" 
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                      tickFormatter={(value) => `$${Math.abs(value).toLocaleString()}`}
                    />
                    <Tooltip
                      contentStyle={{ 
                        backgroundColor: '#1F2937',
                        border: 'none',
                        borderRadius: '0.375rem',
                        color: '#F3F4F6'
                      }}
                    />
                    <Bar 
                      dataKey="value" 
                      fill={(data) => data.value > 0 ? '#34D399' : '#F87171'}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        );
    }
  };

  if (isLoading) return <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">Loading...</div>;
  if (error) return <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6">
      {/* Header with Guide Button */}
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-100 mb-2">Trading Dashboard</h1>
          <p className="text-gray-400">Real-time trading performance and risk analysis</p>
        </div>
        <button
          onClick={() => setShowGuide(true)}
          className="px-4 py-2 bg-blue-600 rounded-lg flex items-center gap-2 hover:bg-blue-500"
        >
          <Info size={20} />
          Beginner's Guide
        </button>
      </div>

      {/* Navigation */}
      <div className="flex space-x-4 mb-8">
        <button 
          onClick={() => setActiveTab('overview')}
          className={`px-4 py-2 rounded ${
            activeTab === 'overview' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-800 text-gray-300'
          }`}
        >
          Overview
        </button>
        <button 
          onClick={() => setActiveTab('positions')}
          className={`px-4 py-2 rounded ${
            activeTab === 'positions' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-800 text-gray-300'
          }`}
        >
          Positions
        </button>
        <button 
          onClick={() => setActiveTab('risk')}
          className={`px-4 py-2 rounded ${
            activeTab === 'risk' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-800 text-gray-300'
          }`}
        >
          Risk Analysis
        </button>
      </div>

      {/* Key Metrics */}
      <div id="daily-metrics" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard 
          title="Win Rate"
          value={`${(data.dailyMetrics.winRate * 100).toFixed(1)}%`}
          icon={<TrendingUp className="h-6 w-6 text-green-400" />}
          trend={data.dailyMetrics.winRateChange >= 0 ? "up" : "down"}
          trendValue={`${(data.dailyMetrics.winRateChange * 100).toFixed(1)}%`}
          onInfoClick={() => setActiveTooltip('winRate')}
          tooltip={activeTooltip === 'winRate' && explanations.winRate}
        />
        {/* Add other metric cards here */}
      </div>

      {/* Tab Content */}
      {renderTabContent()}

      {/* Beginners Guide */}
      <BeginnersGuide 
        isOpen={showGuide}
        onClose={() => setShowGuide(false)}
        currentStep={guideStep}
        onNextStep={setGuideStep}
      />
    </div>
  );
};

export default TradingDashboard;