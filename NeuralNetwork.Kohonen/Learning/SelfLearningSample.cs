using System.Collections.Generic;
using NeuralNetworkConstructor.Learning.Samples;

namespace NeuralNetwork.Kohonen.Learning
{
    public class SelfLearningSample : ISelfLearningSample
    {

        public IEnumerable<double> Input { get; }

        public SelfLearningSample(IEnumerable<double> input) => Input = input;

    }
}