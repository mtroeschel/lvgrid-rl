"""lvgrid-rl: Trainingsumgebung fuer RL-Agenten zur Regelung von Niederspannungsnetzen.

Schichtung (siehe Architekturdokument, Abschnitt 2):

===========  ========================================================
Paket        Schicht
===========  ========================================================
``core``     Schemata, Protokolle, Einheitenkanon, Informationsordnung
``data``     L0 Datenschicht: Quellen, Resampling, Cache, Szenarien
``grid``     L1 Netz und Physik
``components`` L2 Anlagen- und Flexibilitaetsmodelle
``env``      L3 Gymnasium-Environment
``agents``   L4 Agenten
``baselines`` L4 Referenzverfahren
``eval``     L5 KPIs und Auswertung
``viz``      L5 Visualisierung
``experiment`` L6 Orchestrierung und Reproduzierbarkeit
===========  ========================================================
"""

__version__ = "0.1.0.dev0"
