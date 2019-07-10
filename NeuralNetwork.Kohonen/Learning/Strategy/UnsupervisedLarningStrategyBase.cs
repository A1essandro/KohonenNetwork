using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Learning.Strategies;
using NeuralNetwork.Structure.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen.Learning.Strategy
{
    public abstract class UnsupervisedLarningStrategyBase : ILearningStrategy<IKohonenNetwork, ISelfLearningSample>
    {

        public abstract Task LearnSample(IKohonenNetwork network, ISelfLearningSample sample, double theta);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected ISlaveNode GetWinner(IKohonenNetwork network, IEnumerable<double> output, double theta)
        {
            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            return network.OutputLayer.Nodes.ToArray()[winnerIndex] as ISlaveNode;
        }

    }
}