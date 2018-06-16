using System.Collections.Generic;

namespace KohonenNetwork.Learning
{
    public interface ISelfLearning
    {

        void Learn(IEnumerable<double> input);

    }
}