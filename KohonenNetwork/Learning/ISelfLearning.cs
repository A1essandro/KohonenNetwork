using System.Collections.Generic;

namespace KohonenNetwork.Learning
{
    public interface ISelfLearning
    {

        void Learn(ICollection<double> input, double force);

    }
}