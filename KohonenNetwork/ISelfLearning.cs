using System.Collections.Generic;

namespace KohonenNetwork
{
    public interface ISelfLearning
    {

        void Learn(ICollection<double> input, double force);

    }
}