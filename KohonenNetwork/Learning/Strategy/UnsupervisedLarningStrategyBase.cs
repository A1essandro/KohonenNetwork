using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Learning.Samples;
using NeuralNetworkConstructor.Learning.Strategies;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork.Learning.Strategy
{
    public abstract class UnsupervisedLarningStrategyBase : ILearningStrategy<KohonenNetwork, ISelfLearningSample>
    {

        public abstract Task LearnSample(KohonenNetwork network, ISelfLearningSample sample, double theta);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected ISlaveNode GetWinner(KohonenNetwork network, IEnumerable<double> output, double theta)
        {
            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            return network.OutputLayer.Nodes.ToArray()[winnerIndex] as ISlaveNode;
        }

    }
}