/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../../src/control_unit.h"
          

using namespace std;


void Relax(Relaxation* p_rlx1, Relaxation* p_rlx2, Configuration& cfg, double pot_change_treshold)
{
	static int strct_cntr = 1;

	cout << "Relaxation structure#" << strct_cntr
		<< " with " << cfg.size() << " atoms";
	if (p_rlx2 != nullptr)
		cout << " by rough potential. ";
	else
		cout << ". ";
	cout.flush();

	p_rlx1->cfg = cfg;
	p_rlx1->Run();
	cfg = p_rlx1->cfg;
	cout << "Energy after relaxation: " << cfg.energy << ", status: " << cfg.features["from"];
	cout.flush();

	if (p_rlx2 != nullptr && cfg.energy/cfg.size() < pot_change_treshold)
	{
		p_rlx2->struct_cntr = p_rlx1->struct_cntr;
		cout << "\nContinue relaxation structure#"<< strct_cntr << " by fine potential. ";
		p_rlx2->cfg = cfg;
		p_rlx2->Run();
		cfg = p_rlx2->cfg;
		cout << "Energy after relaxation: " << cfg.energy;
	}
	cout << endl;
}


int main(int argc, char* argv[])
{
	try
	{
		if (argc == 0)
			ERROR("wrong argument count");

		int pop_size = stoi(argv[1]);

		ControlUnit cu1("MLIP-settings.ini", true);
		ControlUnit cu2("MLIP-settings_fine.ini", true);
		Relaxation* p_rlx1 = (Relaxation*)cu1.p_driver;
		Relaxation* p_rlx2 = (Relaxation*)cu2.p_driver;

		if (p_rlx1==nullptr)
			ERROR("Relaxation is not initialized");
//		if (p_rlx2==nullptr)
//			p_rlx2->SetLogStream(p_rlx1->GetLogStream());

		for (int i=1; i<=pop_size; i++)
		{
			string fnm = to_string(i) + ".cfg";
			if (system(("mv for_relax/" + fnm + " temp/" + fnm).c_str()) == 0)
			{
				Message("Reading structures from " + fnm);
				ifstream ifs("temp/" + fnm);
				Configuration cfg;
				while (cfg.Load(ifs))
				{
					if (cfg.features.count("pressure") == 0)
						cfg.features["pressure"] = "0";
					p_rlx1->pressure = stod(cfg.features["pressure"]);
					if (p_rlx2 != nullptr)
						p_rlx2->pressure = stod(cfg.features["pressure"]);
					Relax(p_rlx1, p_rlx2, cfg, stod(cu1.settings["USPEX_PotChangeTreshold"]));
					cfg.CorrectSupercell();
					system(("rm -f relaxed/" + fnm).c_str());
					//cfg.features.erase("pressure");
					cfg.AppendToFile("relaxed/" + fnm);
					system(("rm -f temp/" + fnm).c_str());
					Message("Relaxed structure saved to " + fnm);
				}
			}
		}
	}
	catch (MlipException& ex)
	{
		cout << ex.What() << endl;
		cerr << ex.What() << endl;
		return 1;
	}
}

