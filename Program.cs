using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using static AnalisisSentimientos.Sentimiento;

namespace AnalisisSentimientos
{
    // Objetivo: Desarrollar un aplicación ML que nos indique si los comentarios de usuario
    //      son positivos o negativos.
    // Cómo: Algoritmo de clasificación binaria.
    // Datos: Archivo con datos de entrenamiento + archivo con datos de prueba

    internal class Program
    {
        private const string _rutaDatosEntrenamiento = @"..\..\Data\sentiment labelled sentences\imdb_labelled.txt";
        private const string _rutaDatosPrueba = @"..\..\Data\sentiment labelled sentences\yelp_labelled.txt";

        private static void Main(string[] args)
        {
            var modelo = EntrenayPredice();
            Evalua(modelo);
            Console.ReadLine();
        }

        private static PredictionModel<DatosSentimiento, PrediccSentimiento> EntrenayPredice()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<DatosSentimiento>(
                _rutaDatosEntrenamiento,
                useHeader: false,
                separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "Texto"));
            pipeline.Add(new FastTreeBinaryClassifier
            {
                NumLeaves = 5,
                NumTrees = 5,
                MinDocumentsInLeafs = 2
            });

            PredictionModel<DatosSentimiento, PrediccSentimiento> modelo = pipeline
                .Train<DatosSentimiento, PrediccSentimiento>();

            IEnumerable<DatosSentimiento> sentimientos = new[]
            {
                new DatosSentimiento
                {
                    Texto = "This movie was very boring",
                    Etiqueta = 0
                },
                new DatosSentimiento
                {
                    Texto = "This movie did not get my attention",
                    Etiqueta = 0
                },
                new DatosSentimiento
                {
                    Texto = "A super exciting and entertaining movie",
                    Etiqueta = 1
                },
                new DatosSentimiento
                {
                    Texto = "Very good movie"
                },
                new DatosSentimiento
                {
                    Texto = "Worst movie ever"
                }
            };

            var predicciones = modelo.Predict(sentimientos);

            Console.WriteLine();
            Console.WriteLine("Predicción de sentimientos");
            Console.WriteLine("--------------------------");

            var sentimientosYpredicciones = sentimientos.Zip(
                predicciones, (sent, predic) => (sent, predic));
            foreach (var item in sentimientosYpredicciones)
            {
                Console.WriteLine($"Sentimiento: {item.sent.Texto} | Predicción: {(item.predic.Etiqueta ? "Positivo :)" : "Negativo :(")}");
            }
            Console.WriteLine();

            return modelo;
        }

        public static void Evalua(PredictionModel<DatosSentimiento, PrediccSentimiento> modelo)
        {
            var datosPrueba = new TextLoader<DatosSentimiento>(
                _rutaDatosPrueba, useHeader: false, separator: "tab");
            var evaluador = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metricas = evaluador.Evaluate(modelo, datosPrueba);

            Console.WriteLine();
            Console.WriteLine("Evaluación de métricas de calidad del Modelo de Predicción");
            Console.WriteLine("--------------------------");
            Console.WriteLine($"Precisión: {metricas.Accuracy:P2}");
            Console.WriteLine($"AUC: {metricas.Auc:P2}");
            Console.WriteLine($"Entropía: {metricas.Entropy:P2}");
            Console.WriteLine($"LogLoss: {metricas.LogLoss:P2}");
        }
    }
}