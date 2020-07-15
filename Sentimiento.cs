using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;

namespace AnalisisSentimientos
{
    internal class Sentimiento
    {
        public class DatosSentimiento
        {
            [Column("0")]
            public string Texto;

            [Column("1", "Label")]
            public float Etiqueta;
        }

        public class PrediccSentimiento
        {
            [ColumnName("PredictedLabel")]
            public bool Etiqueta;
        }
    }
}