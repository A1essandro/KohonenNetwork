using NeuralNetwork.Learning.Samples;
using System.Collections.Generic;

namespace NeuralNetwork.Kohonen.Learning
{
    public class SelfLearningSample : ISelfLearningSample
    {

        public IEnumerable<double> Input { get; }

        public SelfLearningSample(IEnumerable<double> input) => Input = input;

    }
}