namespace KohonenNetwork.Learning
{
    public class LearningConfiguration
    {

        public const double DEFAULT_THETA = 0.15;

        public double Theta { get; set; }
        public IOrganizing OrganizingAlgorithm { get; set; }
        public double ThetaFactorPerEpoch { get; set; } = 1.0;
        public bool ShuffleEveryEpoch { get; set; } = true;
        public int DefaultRepeatsNumber { get; set; } = 1;

        public LearningConfiguration(double theta = DEFAULT_THETA, IOrganizing organizingAlgorithm = null)
        {
            Theta = theta;
            OrganizingAlgorithm = organizingAlgorithm;
        }

    }
}