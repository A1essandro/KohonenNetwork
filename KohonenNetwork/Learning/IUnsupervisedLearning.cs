using System.Collections.Generic;
using NeuralNetworkConstructor.Network;

namespace KohonenNetwork.Learning
{
    public interface IUnsupervisedLearning
    {

        void Learn(IEnumerable<double> input);

        void SetNetwork(INetwork network);

    }
}