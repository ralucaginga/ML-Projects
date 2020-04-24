public class ipm {
	
	public static float InteriorPoint(float[][] G, float[][] c, float[][] A, float[][] b, float[][] x, float[][] y, float[][] lambda){
		int m = A.length; // no. of rows
		int n = A[0].length; // no. of columns
		int maxIteration = 100;
		float[][] tau = new float[maxIteration][1];
		for(int i = 0; i < maxIteration; i++)
			tau[i][0] = 0f;
		float zstar;
		float[][] zstar1 = new float[maxIteration + 1][1];
		zstar1[0][0] = findValue(G, x);
		
		for(int k = 0; k < maxIteration; k++) {
			float[][] rd = new float[G.length][1];
			float[][] rp = new float[A.length][1];
			rd = computeRD(G, x, A, lambda, c);
			rp = computeRP(A, x, y, b);
			int dimension = G.length + 2 * m;
			float[][] F = new float[dimension][dimension];
			
			// Populating the F matrix
			for(int i = 0; i < G.length; i++)
				for(int j = 0; j < G.length; j++)
					F[i][j] = G[i][j];
			for(int i = 0; i < G.length; i++)
				for(int j = G.length; j < G.length + m; j++)
					F[i][j] = 0;
			float[][] trMinusA = transposeMatrix(A);
			int z = 0;
			for(int i = 0; i < G.length; i++) {
				int yy = 0;
				for(int j = G.length + m; j < dimension; j++) {
					F[i][j] = (-1) * trMinusA[z][yy];
					yy += 1;
				}
				z += 1;	
			}
			int zz = 0;
			for(int i = G.length; i < G.length + m; i++) {
				int yyy = 0;
				for(int j = 0; j < n; j++) {
					F[i][j] = A[zz][yyy];
					yyy += 1;
				}
				zz += 1;	
			}
			for(int i = G.length; i < G.length + m; i++) {
				for(int j = G.length; j < G.length + m; j++) 
					if(i == j)
						F[i][j] = -1;
					else F[i][j] = 0;
			}
			for(int i = G.length; i < G.length + m; i++)
				for(int j = G.length + m; j < dimension; j++)
					F[i][j] = 0;
			for(int i = G.length + m; i < dimension; i++)
				for(int j = 0; j < n; j++)
					F[i][j] = 0;
			int zzz = 0;
			float[][] diaglambda = diag(lambda);
			for(int i = G.length + m; i < dimension; i++) {
				for(int j = G.length; j < G.length + m; j++) {
					if(i - j == m)
						F[i][j] = diaglambda[zzz][zzz];
				}
				zzz += 1;
			}
			int zzzz = 0;
			float[][] diagy = diag(y);
			for(int i = G.length + m; i < dimension; i++) {
				for(int j = G.length + m; j < dimension; j++) {
					if(i == j)
						F[i][j] = diagy[zzzz][zzzz];
				}
				zzzz += 1;
			}
			// Populating the equation_b matrix
			float[][] equation_b = new float[dimension][1];
			for(int i = 0; i < rd.length; i++)
				equation_b[i][0] = (-1) * rd[i][0];
			int mm = 0;
			for(int i = rd.length; i < rd.length + rp.length; i++) {
				equation_b[i][0] = (-1) * rp[mm][0];
				mm++;
			}
			
			float[][] ones = new float[m][1];
			for(int i = 0; i < m; i++)
				ones[i][0] = 1;
			float[][] multiplyDiagLbdYOnes = multiplyMatrix(multiplyMatrixMinus(diaglambda, diagy), ones);
			int mo = 0;
			for(int i = rd.length + rp.length; i < dimension; i++) {
				equation_b[i][0] = multiplyDiagLbdYOnes[mo][0];
				mo++;
			}
			
			float[] b1 = new float[equation_b.length];
			for(int i = 0; i < b1.length; i++)
				b1[i] = equation_b[i][0];
			float[] solution = GaussJordanElimination.test(F, b1);
			float[][] delta_x_aff = new float[n][1];
			float[][] delta_y_aff = new float[m][1];
			float[][] delta_lambda_aff = new float[m][1];
			// Populating delta_x_aff matrix
			for(int i = 0; i < n; i++)
				delta_x_aff[i][0] = 0;
			// Populating delta_y_aff matrix
			for(int i = 0; i < m; i++)
				delta_y_aff[i][0] = 0;
			// Populating delta_lambda_aff matrix
			for(int i = 0; i < m; i++)
				delta_lambda_aff[i][0] = 0;
			
			for(int kk = 0; kk < n; kk++)
				delta_x_aff[kk][0] = solution[kk];
			for(int kk = n; kk < n + m; kk++)
				delta_y_aff[kk - n][0] = solution[kk];
			for(int kk = n + m; kk < n + 2 * m; kk++)
				delta_lambda_aff[kk - n - m][0] = solution[kk];
		
			// Computing mu
			float[][] muM = multiplyMatrix(transposeMatrix(y), lambda);
			float mu = muM[0][0] / m;
			
			float alpha_aff = 1;
			float max1 = maxSumLessThan0(y, alpha_aff, delta_y_aff);
			float max2 = maxSumLessThan0(lambda, alpha_aff, delta_lambda_aff);
			boolean compare = compareTwoMax(max1, max2);
			while(compareTwoMax(max1, max2)) {
				alpha_aff -= 0.01f;
				if(alpha_aff <= 0)
					break;
				max1 = maxSumLessThan0(y, alpha_aff, delta_y_aff);
				max2 = maxSumLessThan0(lambda, alpha_aff, delta_lambda_aff);
			}
			
			float[][] sumyalphadelta = computeSum(y, alpha_aff, delta_y_aff);
			float[][] sumlambdaalphadelta = computeSum(lambda, alpha_aff, delta_lambda_aff);
			float[][] muA = multiplyMatrix(transposeMatrix(sumyalphadelta),sumlambdaalphadelta);
			float mu_aff = muA[0][0] / m;
			
			float sigma = (mu_aff / mu) * (mu_aff / mu) * (mu_aff / mu);
			for(int i = 0; i < rd.length; i++)
				equation_b[i][0] = (-1) * rd[i][0];
			mm = 0;
			for(int i = rd.length; i < rd.length + rp.length; i++) {
				equation_b[i][0] = (-1) * rp[mm][0];
				mm++;
			}
			float[][] diagLambdaAff = diag(delta_lambda_aff);
			float[][] diagYAff = diag(delta_y_aff);
			float[][] scalars = ScalarsMatrix(sigma, mu, ones);
			float[][] multiplyDiagLbdAffYOnes = multiplyMatrix(multiplyMatrixMinus(diagLambdaAff, diagYAff), ones);
			float[][] addedMatrix = addMatrix(addMatrix(multiplyDiagLbdYOnes, multiplyDiagLbdAffYOnes),scalars);
			
			mo = 0;
			for(int i = rd.length + rp.length; i < dimension; i++) {
				equation_b[i][0] = addedMatrix[mo][0];
				mo++;
			}
			
			float[][] delta_x = new float[n][1];
			for(int i = 0; i < n; i++)
				delta_x[i][0] = 0;
			float[][] delta_y = new float[m][1];
			for(int i = 0; i < m; i++)
				delta_y[i][0] = 0;
			float[][] delta_lambda = new float[m][1];
			for(int i = 0; i < m; i++)
				delta_lambda[i][0] = 0;
			
			for(int i = 0; i < b1.length; i++)
				b1[i] = equation_b[i][0];
			solution = GaussJordanElimination.test(F, b1);
			
			for(int kk = 0; kk < n; kk++)
				delta_x[kk][0] = solution[kk];
			for(int kk = n; kk < n + m; kk++)
				delta_y[kk - n][0] = solution[kk];
			for(int kk = n + m; kk < n + 2 * m; kk++)
				delta_lambda[kk - n - m][0] = solution[kk];
			
			tau[k][0] = 0.6f;
			float alpha_tau_pri = 1;
			float[][] yAlphaTauDelta = computeSum(y, alpha_tau_pri, delta_y);
			float[][] oneMinusY = oneMinusTau(tau[k][0], y);
			float[][] temp_cond = compareIneq(yAlphaTauDelta, oneMinusY);
			while(sumCompareIneq(temp_cond)) {
				alpha_tau_pri -= 0.01f;
				if(alpha_tau_pri <= 0)
					break;
				yAlphaTauDelta = computeSum(y, alpha_tau_pri, delta_y);
				temp_cond = compareIneq(yAlphaTauDelta, oneMinusY);
			}
			float alpha_tau_dual = 1;
			float[][] lambdaAlphaDTauDelta = computeSum(lambda, alpha_tau_dual, delta_lambda);
			float[][] oneMinusLambda = oneMinusTau(tau[k][0], lambda);
			temp_cond = compareIneq(lambdaAlphaDTauDelta, oneMinusLambda);
			while(sumCompareIneq(temp_cond)) {
				alpha_tau_dual -= 0.01f;
				if(alpha_tau_dual <= 0)
					break;
				lambdaAlphaDTauDelta = computeSum(lambda, alpha_tau_dual, delta_lambda);
				temp_cond = compareIneq(lambdaAlphaDTauDelta, oneMinusLambda);
			}
			
			float alpha = Math.min(alpha_tau_pri, alpha_tau_dual);
			x = computeX(x, alpha, delta_x);
			y = computeX(y, alpha, delta_y);
			lambda = computeX(lambda, alpha, delta_lambda);
			
			float[][] zstea = multiplyMatrix(multiplyMatrix(transposeMatrix(x),G), x);
			zstea[0][0] *= 0.5;
			zstar1[k+1][0] = zstea[0][0];
			
			if(Math.abs(zstar1[k+1][0] - zstar1[k][0]) < 1e-8f)
				break;
		}
		// Solution of the problem
		float[][] sol = new float[x.length][1];
		for(int i = 0; i < sol.length; i++)
			sol[i][0] = x[i][0];
		
		// Minimum value of the problem
		float[][] zstar2 = multiplyMatrix(multiplyMatrix(transposeMatrix(x),G), x);
		zstar = zstar2[0][0] * 0.5f;
		return zstar;
	}
	
	public static float[][] IPM(float[][] G, float[][] c, float[][] A, float[][] b, float[][] x, float[][] y, float[][] lambda){
		int m = A.length; // no. of rows
		int n = A[0].length; // no. of columns
		int maxIteration = 100;
		float[][] tau = new float[maxIteration][1];
		for(int i = 0; i < maxIteration; i++)
			tau[i][0] = 0f;
		float zstar;
		float[][] zstar1 = new float[maxIteration + 1][1];
		zstar1[0][0] = findValue(G, x);
		
		for(int k = 0; k < maxIteration; k++) {
			float[][] rd = new float[G.length][1];
			float[][] rp = new float[A.length][1];
			rd = computeRD(G, x, A, lambda, c);
			rp = computeRP(A, x, y, b);
			int dimension = G.length + 2 * m;
			float[][] F = new float[dimension][dimension];
			
			// Populating the F matrix
			for(int i = 0; i < G.length; i++)
				for(int j = 0; j < G.length; j++)
					F[i][j] = G[i][j];
			for(int i = 0; i < G.length; i++)
				for(int j = G.length; j < G.length + m; j++)
					F[i][j] = 0;
			float[][] trMinusA = transposeMatrix(A);
			int z = 0;
			for(int i = 0; i < G.length; i++) {
				int yy = 0;
				for(int j = G.length + m; j < dimension; j++) {
					F[i][j] = (-1) * trMinusA[z][yy];
					yy += 1;
				}
				z += 1;	
			}
			int zz = 0;
			for(int i = G.length; i < G.length + m; i++) {
				int yyy = 0;
				for(int j = 0; j < n; j++) {
					F[i][j] = A[zz][yyy];
					yyy += 1;
				}
				zz += 1;	
			}
			for(int i = G.length; i < G.length + m; i++) {
				for(int j = G.length; j < G.length + m; j++) 
					if(i == j)
						F[i][j] = -1;
					else F[i][j] = 0;
			}
			for(int i = G.length; i < G.length + m; i++)
				for(int j = G.length + m; j < dimension; j++)
					F[i][j] = 0;
			for(int i = G.length + m; i < dimension; i++)
				for(int j = 0; j < n; j++)
					F[i][j] = 0;
			int zzz = 0;
			float[][] diaglambda = diag(lambda);
			for(int i = G.length + m; i < dimension; i++) {
				for(int j = G.length; j < G.length + m; j++) {
					if(i - j == m)
						F[i][j] = diaglambda[zzz][zzz];
				}
				zzz += 1;
			}
			int zzzz = 0;
			float[][] diagy = diag(y);
			for(int i = G.length + m; i < dimension; i++) {
				for(int j = G.length + m; j < dimension; j++) {
					if(i == j)
						F[i][j] = diagy[zzzz][zzzz];
				}
				zzzz += 1;
			}
			// Populating the equation_b matrix
			float[][] equation_b = new float[dimension][1];
			for(int i = 0; i < rd.length; i++)
				equation_b[i][0] = (-1) * rd[i][0];
			int mm = 0;
			for(int i = rd.length; i < rd.length + rp.length; i++) {
				equation_b[i][0] = (-1) * rp[mm][0];
				mm++;
			}
			
			float[][] ones = new float[m][1];
			for(int i = 0; i < m; i++)
				ones[i][0] = 1;
			float[][] multiplyDiagLbdYOnes = multiplyMatrix(multiplyMatrixMinus(diaglambda, diagy), ones);
			int mo = 0;
			for(int i = rd.length + rp.length; i < dimension; i++) {
				equation_b[i][0] = multiplyDiagLbdYOnes[mo][0];
				mo++;
			}
			
			float[] b1 = new float[equation_b.length];
			for(int i = 0; i < b1.length; i++)
				b1[i] = equation_b[i][0];
			float[] solution = GaussJordanElimination.test(F, b1);
			float[][] delta_x_aff = new float[n][1];
			float[][] delta_y_aff = new float[m][1];
			float[][] delta_lambda_aff = new float[m][1];
			// Populating delta_x_aff matrix
			for(int i = 0; i < n; i++)
				delta_x_aff[i][0] = 0;
			// Populating delta_y_aff matrix
			for(int i = 0; i < m; i++)
				delta_y_aff[i][0] = 0;
			// Populating delta_lambda_aff matrix
			for(int i = 0; i < m; i++)
				delta_lambda_aff[i][0] = 0;
			
			for(int kk = 0; kk < n; kk++)
				delta_x_aff[kk][0] = solution[kk];
			for(int kk = n; kk < n + m; kk++)
				delta_y_aff[kk - n][0] = solution[kk];
			for(int kk = n + m; kk < n + 2 * m; kk++)
				delta_lambda_aff[kk - n - m][0] = solution[kk];
		
			// Computing mu
			float[][] muM = multiplyMatrix(transposeMatrix(y), lambda);
			float mu = muM[0][0] / m;
			
			float alpha_aff = 1;
			float max1 = maxSumLessThan0(y, alpha_aff, delta_y_aff);
			float max2 = maxSumLessThan0(lambda, alpha_aff, delta_lambda_aff);
			boolean compare = compareTwoMax(max1, max2);
			while(compareTwoMax(max1, max2)) {
				alpha_aff -= 0.01f;
				if(alpha_aff <= 0)
					break;
				max1 = maxSumLessThan0(y, alpha_aff, delta_y_aff);
				max2 = maxSumLessThan0(lambda, alpha_aff, delta_lambda_aff);
			}
			
			float[][] sumyalphadelta = computeSum(y, alpha_aff, delta_y_aff);
			float[][] sumlambdaalphadelta = computeSum(lambda, alpha_aff, delta_lambda_aff);
			float[][] muA = multiplyMatrix(transposeMatrix(sumyalphadelta),sumlambdaalphadelta);
			float mu_aff = muA[0][0] / m;
			
			float sigma = (mu_aff / mu) * (mu_aff / mu) * (mu_aff / mu);
			for(int i = 0; i < rd.length; i++)
				equation_b[i][0] = (-1) * rd[i][0];
			mm = 0;
			for(int i = rd.length; i < rd.length + rp.length; i++) {
				equation_b[i][0] = (-1) * rp[mm][0];
				mm++;
			}
			float[][] diagLambdaAff = diag(delta_lambda_aff);
			float[][] diagYAff = diag(delta_y_aff);
			float[][] scalars = ScalarsMatrix(sigma, mu, ones);
			float[][] multiplyDiagLbdAffYOnes = multiplyMatrix(multiplyMatrixMinus(diagLambdaAff, diagYAff), ones);
			float[][] addedMatrix = addMatrix(addMatrix(multiplyDiagLbdYOnes, multiplyDiagLbdAffYOnes),scalars);
			
			mo = 0;
			for(int i = rd.length + rp.length; i < dimension; i++) {
				equation_b[i][0] = addedMatrix[mo][0];
				mo++;
			}
			
			float[][] delta_x = new float[n][1];
			for(int i = 0; i < n; i++)
				delta_x[i][0] = 0;
			float[][] delta_y = new float[m][1];
			for(int i = 0; i < m; i++)
				delta_y[i][0] = 0;
			float[][] delta_lambda = new float[m][1];
			for(int i = 0; i < m; i++)
				delta_lambda[i][0] = 0;
			
			for(int i = 0; i < b1.length; i++)
				b1[i] = equation_b[i][0];
			solution = GaussJordanElimination.test(F, b1);
			
			for(int kk = 0; kk < n; kk++)
				delta_x[kk][0] = solution[kk];
			for(int kk = n; kk < n + m; kk++)
				delta_y[kk - n][0] = solution[kk];
			for(int kk = n + m; kk < n + 2 * m; kk++)
				delta_lambda[kk - n - m][0] = solution[kk];
			
			tau[k][0] = 0.6f;
			float alpha_tau_pri = 1;
			float[][] yAlphaTauDelta = computeSum(y, alpha_tau_pri, delta_y);
			float[][] oneMinusY = oneMinusTau(tau[k][0], y);
			float[][] temp_cond = compareIneq(yAlphaTauDelta, oneMinusY);
			while(sumCompareIneq(temp_cond)) {
				alpha_tau_pri -= 0.01f;
				if(alpha_tau_pri <= 0)
					break;
				yAlphaTauDelta = computeSum(y, alpha_tau_pri, delta_y);
				temp_cond = compareIneq(yAlphaTauDelta, oneMinusY);
			}
			float alpha_tau_dual = 1;
			float[][] lambdaAlphaDTauDelta = computeSum(lambda, alpha_tau_dual, delta_lambda);
			float[][] oneMinusLambda = oneMinusTau(tau[k][0], lambda);
			temp_cond = compareIneq(lambdaAlphaDTauDelta, oneMinusLambda);
			while(sumCompareIneq(temp_cond)) {
				alpha_tau_dual -= 0.01f;
				if(alpha_tau_dual <= 0)
					break;
				lambdaAlphaDTauDelta = computeSum(lambda, alpha_tau_dual, delta_lambda);
				temp_cond = compareIneq(lambdaAlphaDTauDelta, oneMinusLambda);
			}
			
			float alpha = Math.min(alpha_tau_pri, alpha_tau_dual);
			x = computeX(x, alpha, delta_x);
			y = computeX(y, alpha, delta_y);
			lambda = computeX(lambda, alpha, delta_lambda);
			
			float[][] zstea = multiplyMatrix(multiplyMatrix(transposeMatrix(x),G), x);
			zstea[0][0] *= 0.5;
			zstar1[k+1][0] = zstea[0][0];
			
			if(Math.abs(zstar1[k+1][0] - zstar1[k][0]) < 1e-8f)
				break;
		}
		// Solution of the problem
		float[][] sol = new float[x.length][1];
		for(int i = 0; i < sol.length; i++)
			sol[i][0] = x[i][0];
		
		return sol;
	}
	
	public static float[][] transposeMatrix(float[][] x){
		float[][] transpose = new float[x[0].length][x.length];
		for(int i = 0; i < x[0].length; i++)
			for(int j = 0; j < x.length; j++)
				transpose[i][j] = x[j][i];
		return transpose;
	}

	
	public static float findValue(float[][] G, float[][] x){
        float[][] tx = new float[x.length][1];
        for(int i = 0; i < x.length;i++){
              tx[i][0] = x[i][0];
        }
        float number = 0;
        for(int i = 0; i < G[0].length; i++){
                 for(int j = 0; j < x.length; j++){
                            number += (1/2)  * tx[j][0] * G[i][j] * x[j][0];
        	     }
        }
        return number;   	
	}
	
	public static float[][] computeX(float[][] x, float alpha, float[][] delta){
		float[][] compute = new float[x.length][1];
		for(int i = 0; i < x.length; i++)
			compute[i][0] = x[i][0] + alpha * delta[i][0];
		return compute;
	}
	public static float[][] oneMinusTau(float tau, float[][] y){
		float[][] compute = new float[y.length][1];
		for(int i = 0; i < y.length; i++)
			compute[i][0] = (1 - tau) * y[i][0];
		return compute;
	}
	
	public static float[][] compareIneq(float[][] first, float[][] second) {
		float[][] compute = new float[first.length][1];
		for(int i = 0; i < first.length; i++)
			if(first[i][0] <= second[i][0])
				compute[i][0] = 1;
			else compute[i][0] = 0;
		return compute;
	}
	
	public static boolean sumCompareIneq(float[][] first) {
		float sum = 0;
		for(int i = 0; i < first.length; i++)
			sum += first[i][0];
		
		if(sum > 0)
			return true;
		else 
			return false;
	}
	
	public static float[][] addMatrix(float[][] A, float[][] B){
		float[][] compute = new float[A.length][A[0].length];
		for(int i = 0; i < A.length; i++) 
			for(int j = 0; j < A[0].length; j++)
				compute[i][j] = A[i][j] + B[i][j];
		return compute;
	}
	
	public static float[][] computeRD(float[][] G, float[][] x, float[][] A, float[][] lambda, float[][] c){
		float[][] compute = new float[G.length][1];
		float[][] transposeA = transposeMatrix(A);
		for(int i = 0; i < G.length; i++){
            float Gx = 0;
            float Alambda = 0;
            for(int j = 0; j < G[0].length; j++)
                Gx += G[i][j] * x[j][0];
            for(int k = 0; k < A.length; k++)
            	Alambda += transposeA[i][k] * lambda[k][0];
            compute[i][0] = Gx - Alambda + c[i][0];
        }
        return compute;
	}
	
	public static float[][] computeRP(float[][] A, float[][] x, float[][] y, float[][] b){
		float[][] compute = new float[A.length][1];
		for(int i = 0; i < A.length; i++){
            float Ax = 0;
            for(int j = 0; j < A[0].length; j++)
                Ax += A[i][j] * x[j][0];
            compute[i][0] = Ax - y[i][0] - b[i][0];
        }
        return compute;
	}
	
	public static float[][] diag(float[][] a){
		float[][] dia = new float[a.length][a.length];
		for(int i = 0; i < a.length; i++)
			for(int j = 0; j < a.length; j++)
				if(i==j)
					dia[i][j] = a[i][0];
		return dia;
	}
	
	public static float[][] multiplyMatrixMinus(float[][] A, float[][] B){
		float[][] C = new float[A.length][B[0].length];
		for(int i = 0; i < A.length; i++)
			for(int j = 0; j < B[0].length; j++)
				for(int k = 0; k < B.length; k++)
				C[i][j] += (-1) * A[i][k] * B[k][j];
		return C;
	}
	
	public static float[][] multiplyMatrix(float[][] A, float[][] B){
		float[][] C = new float[A.length][B[0].length];
		for(int i = 0; i < A.length; i++)
			for(int j = 0; j < B[0].length; j++)
				for(int k = 0; k < B.length; k++)
				C[i][j] += A[i][k] * B[k][j];
		return C;
	}
	
	public static float[][] computeSum(float[][] y, float alpha, float[][] delta){
		float[][] compute = new float[y.length][1];
		for(int i = 0; i < y.length; i++) 
			compute[i][0] = y[i][0] + alpha * delta[i][0];
		return compute;
	}
	
	public static float[][] ScalarsMatrix(float a, float b, float[][] ones){
		float[][] compute = new float[ones.length][1];
		for(int i = 0; i < ones.length; i++)
			compute[i][0] = a * b * ones[i][0];
		return compute;
	}
	
	public static float maxSumLessThan0(float[][] y, float alpha, float[][] delta){
		float[][] compute = new float[y.length][1];
		for(int i = 0; i < y.length; i++) 
			compute[i][0] = y[i][0] + alpha * delta[i][0];
		for(int j = 0; j < y.length; j++)
			if(compute[j][0] < 0)
				compute[j][0] = 1;
			else compute[j][0] = 0;
		float maxVector = 0;
		for(int i = 0; i < y.length; i++)
			if(maxVector <= compute[i][0])
				maxVector = compute[i][0];
		return maxVector;
	}
	
	public static boolean compareTwoMax(float m1, float m2) {
		if(m1 == 0)
			if(m2 == 1)
				return true;
			else return false;
		if(m2 == 0)
			if(m1 == 1)
				return true;
			else return false;
		return true;
	}
    
	
	public static void main(String[] args){
        float[][] G = new float[][]{{1,0},{0,1}};
        float[][] c = new float[][]{{0},{0}};
        float[][] A = new float[][]{{100,100},{-55,-55}};
        float[][] b = new float[][]{{5},{-3}};
        float[][] x = new float[][] {{0},{0}};
        float[][] y = new float[][] {{1},{1}};
        float[][] lambda = new float[][] {{1},{1}};
        
       float solution = InteriorPoint(G, c, A, b, x, y, lambda);
       System.out.println(solution);
       
       float[][] solutione = IPM(G, c, A, b, x, y, lambda);
       for(int i = 0; i < 2; i++)
    	   System.out.println(solutione[i][0]);
	}
}

// https://introcs.cs.princeton.edu/java/95linear/GaussJordanElimination.java.html
class GaussJordanElimination {
    private static final float EPSILON = 1e-8f;

    private final int n;      // n-by-n system
    private float[][] a;     // n-by-(n+1) augmented matrix

    // Gauss-Jordan elimination with partial pivoting
    /**
     * Solves the linear system of equations <em>Ax</em> = <em>b</em>,
     * where <em>A</em> is an <em>n</em>-by-<em>n</em> matrix and <em>b</em>
     * is a length <em>n</em> vector.
     *
     * @param  a2 the <em>n</em>-by-<em>n</em> constraint matrix
     * @param  b the length <em>n</em> right-hand-side vector
     */
    public GaussJordanElimination(float[][] a2, float[] b) {
        n = b.length;

        // build augmented matrix
        a = new float[n][n+n+1];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = a2[i][j];

        // only needed if you want to find certificate of infeasibility (or compute inverse)
        for (int i = 0; i < n; i++)
            a[i][n+i] = 1.0f;

        for (int i = 0; i < n; i++)
            a[i][n+n] = b[i];

        solve();

        assert certifySolution(a2, b);
    }

    public void solve() {

        // Gauss-Jordan elimination
        for (int p = 0; p < n; p++) {
            // show();

            // find pivot row using partial pivoting
            int max = p;
            for (int i = p+1; i < n; i++) {
                if (Math.abs(a[i][p]) > Math.abs(a[max][p])) {
                    max = i;
                }
            }

            // exchange row p with row max
            swap(p, max);

            // singular or nearly singular
            if (Math.abs(a[p][p]) <= EPSILON) {
                continue;
                // throw new ArithmeticException("Matrix is singular or nearly singular");
            }

            // pivot
            pivot(p, p);
        }
        // show();
    }

    // swap row1 and row2
    public void swap(int row1, int row2) {
        float[] temp = a[row1];
        a[row1] = a[row2];
        a[row2] = temp;
    }


    // pivot on entry (p, q) using Gauss-Jordan elimination
    public void pivot(int p, int q) {

        // everything but row p and column q
        for (int i = 0; i < n; i++) {
            float alpha = a[i][q] / a[p][q];
            for (int j = 0; j <= n+n; j++) {
                if (i != p && j != q) a[i][j] -= alpha * a[p][j];
            }
        }

        // zero out column q
        for (int i = 0; i < n; i++)
            if (i != p) a[i][q] = 0.0f;

        // scale row p (ok to go from q+1 to n, but do this for consistency with simplex pivot)
        for (int j = 0; j <= n+n; j++)
            if (j != q) a[p][j] /= a[p][q];
        a[p][q] = 1.0f;
    }

    /**
     * Returns a solution to the linear system of equations <em>Ax</em> = <em>b</em>.
     *      
     * @return a solution <em>x</em> to the linear system of equations
     *         <em>Ax</em> = <em>b</em>; {@code null} if no such solution
     */
    public float[] primal() {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            if (Math.abs(a[i][i]) > EPSILON)
                x[i] = a[i][n+n] / a[i][i];
            else if (Math.abs(a[i][n+n]) > EPSILON)
                return null;
        }
        return x;
    }

    /**
     * Returns a solution to the linear system of equations <em>yA</em> = 0,
     * <em>yb</em> &ne; 0.
     *      
     * @return a solution <em>y</em> to the linear system of equations
     *         <em>yA</em> = 0, <em>yb</em> &ne; 0; {@code null} if no such solution
     */
    public float[] dual() {
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            if ((Math.abs(a[i][i]) <= EPSILON) && (Math.abs(a[i][n+n]) > EPSILON)) {
                for (int j = 0; j < n; j++)
                    y[j] = a[i][n+j];
                return y;
            }
        }
        return null;
    }

    /**
     * Returns true if there exists a solution to the linear system of
     * equations <em>Ax</em> = <em>b</em>.
     *      
     * @return {@code true} if there exists a solution to the linear system
     *         of equations <em>Ax</em> = <em>b</em>; {@code false} otherwise
     */
    public boolean isFeasible() {
        return primal() != null;
    }

    // print the tableaux
    public void show() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.println("%8.3f "+ a[i][j]);
            }
            System.out.println("| ");
            for (int j = n; j < n+n; j++) {
            	System.out.println("%8.3f "+ a[i][j]);
            }
            System.out.println("| %8.3f\n" + a[i][n+n]);
        }
        System.out.println();
    }


    // check that Ax = b or yA = 0, yb != 0
    public boolean certifySolution(float[][] A, float[] b) {

        // check that Ax = b
        if (isFeasible()) {
            float[] x = primal();
            for (int i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    sum += A[i][j] * x[j];
                }
                if (Math.abs(sum - b[i]) > EPSILON) {
                	System.out.println("not feasible");
                	System.out.println("b[%d] = %8.3f, sum = %8.3f\n" + i + b[i] + sum);
                    return false;
                }
            }
            return true;
        }

        // or that yA = 0, yb != 0
        else {
            float[] y = dual();
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    sum += A[i][j] * y[i];
                }
                if (Math.abs(sum) > EPSILON) {
                	System.out.println("invalid certificate of infeasibility");
                	System.out.println("sum = %8.3f\n" + sum);
                    return false;
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += y[i] * b[i];
            }
            if (Math.abs(sum) < EPSILON) {
            	System.out.println("invalid certificate of infeasibility");
            	System.out.println("yb  = %8.3f\n" + sum);
                return false;
            }
            return true;
        }
    }

    public static float[] test(float[][] A, float[] b) {
    	GaussJordanElimination gaussian = new GaussJordanElimination(A, b);
        if (gaussian.isFeasible()) {
        	//System.out.println("Solution to Ax = b");
            float[] x = gaussian.primal();
            return x;
        }
        else {
        	//System.out.println("Certificate of infeasibility");
            float[] y = gaussian.dual();
            return y;
        }
    }

    public static void test(String name, float[][] A, float[] b) {
    	System.out.println("----------------------------------------------------");
    	System.out.println(name);
    	System.out.println("----------------------------------------------------");
        GaussJordanElimination gaussian = new GaussJordanElimination(A, b);
        if (gaussian.isFeasible()) {
        	System.out.println("Solution to Ax = b");
            float[] x = gaussian.primal();
            for (int i = 0; i < x.length; i++) {
            	System.out.print(x[i] + " ");
            }
        }
        else {
        	System.out.println("Certificate of infeasibility");
            float[] y = gaussian.dual();
            for (int j = 0; j < y.length; j++) {
            	System.out.println(y[j] + " ");
            }
        }
        System.out.println();
        System.out.println();
    }
}
